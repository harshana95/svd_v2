from datetime import datetime
import gc
import importlib
import logging
import math
import os
from pathlib import Path
import sys
import timeit
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel
from huggingface_hub import create_repo, hf_hub_download, upload_folder, delete_repo, repo_exists, whoami

from torch.utils.data import DataLoader, Subset
from diffusers import get_scheduler
from tqdm import tqdm
from utils import get_dataset_util, initialize, keep_last_checkpoints
from utils.dataset_utils import DictWrapper,  patchify
from utils.misc import find_attr, scandir

from models.archs import _arch_modules
from dataset import create_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir):
    i = len(weights) - 1
    saved = {}
    while len(weights) > 0:
        weights.pop()
        model = models[i]

        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        model.save_pretrained(os.path.join(output_dir, f"{class_name}_{saved[class_name]}"))

        i -= 1

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        try:
            c = find_attr(_arch_modules, class_name)
        except ValueError as e:  # class is not written by us. Try to load from diffusers
            print(f"Class {class_name} not found in archs. Trying to load from diffusers...")
            m = importlib.import_module('diffusers') # load the module, will raise ImportError if module cannot be loaded
            c = getattr(m, class_name)  # get the class, will raise AttributeError if class cannot be found    
        
        assert c is not None
        assert c.config_class is not None
        # load diffusers style into model
        try:
            load_model = c.from_pretrained(os.path.join(input_dir, f"{class_name}_{saved[class_name]}"))
            model.load_state_dict(load_model.state_dict())
            del load_model
        except Exception as e:
            print(f"{'='*50} Failed to load {class_name} {'='*50} {e} {c} {model}")

class BaseModel():
    """Base model."""

    def __init__(self, opt, logger=None):
        self.experiment_name = opt.experiment_key+f" {datetime.now().strftime('%Y %m %d %H.%M.%S')}"
        self.accelerator = initialize(opt, logger, self.experiment_name)
        logger.info(opt)
        self.opt = opt
        self.logger = logger
        
        self.save_handle = self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.models = []
        self.overrode_max_train_steps = False
        self.global_step = 0
        self.max_val_steps = opt.val.get('max_val_steps', 1000)

        # enable disable dataset caching
        if opt.dataset_caching or opt.dataset_caching is None:
            from datasets import enable_caching
            enable_caching()
        else:
            from datasets import disable_caching
            disable_caching()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if opt.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if opt.train.optim.scale_lr:  # TODO: Check setting opt values works properly
            opt.train.optim.learning_rate = (
                    opt.train.optim.learning_rate * opt.train.gradient_accumulation_steps * opt.train.batch_size * self.accelerator.num_processes
            )
        self.trackers_initialized = False

    def prepare_trackers(self):
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_kwargs = self.opt.tracker_kwargs
            self.logger.info(f"tracker_kwargs: {tracker_kwargs}")
            # tensorboard cannot handle list types for config
            if tracker_kwargs is None:
                tracker_kwargs = {}
            self.accelerator.init_trackers(
                self.opt.tracker_project_name, 
                init_kwargs=tracker_kwargs, 
            )
            for tracker in self.accelerator.trackers:
                try:
                    tracker.writer.set_name(self.experiment_name)
                    tracker.writer.log_parameters(self.opt)
                    tracker.writer.log_asset(self.opt.opt_path, file_name="config.yml", overwrite=True)
                except:
                    pass
        self.trackers_initialized = True
                
    def check_model_speed(self):
        iters = len(self.dataloader)
        wait = 1
        warmup = 1
        active = 1
        repeat = 1
        n = (wait + warmup + active) * repeat
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t = []
        exp_root = self.opt.path.experiments_root
        
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(exp_root, self.opt.path.logging_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            pbar = tqdm(total=n, desc='Profiling ')
            for i_batch, sample_batched in enumerate(self.dataloader):
                # for _ in range(n):
                start.record()
                
                self.feed_data(sample_batched, is_train=True)
                if self.opt.train.patched:
                    for _ in self.setup_patches():
                        losses = self.optimize_parameters()
                else:
                    losses = self.optimize_parameters()
                
                for i in range(len(self.schedulers)):
                    self.schedulers[i].step()

                if i_batch % 3 == 0:
                    end.record()
                    torch.cuda.synchronize()
                    t.append(start.elapsed_time(end))
                    prof.step()
                pbar.update()
                if i_batch > n:
                    break
            pbar.close()
            print(f"Saved profiler to {os.path.join(exp_root, self.opt.path.logging_dir)}")    
                
    def preview_datasets(self, max_images=10):
        def save_images(sample, name):
            for key in sample.keys():
                if type(sample[key]) == list:
                    sample[key] = np.array(sample[key])
                print(f"Data {name} {str(sample[key].dtype):15}{key:20}: {sample[key].shape}")
            if len(sample[self.gt_key].shape) == 5:
                sample[self.gt_key] = einops.rearrange(sample[self.gt_key], "b n c h w -> (b n) c h w")
            if len(sample[self.blur_key].shape) == 5:
                sample[self.blur_key] = einops.rearrange(sample[self.blur_key], "b n c h w -> (b n) c h w")
            for _i in range(len(sample[self.gt_key])):
                if _i == max_images:
                    break
                plt.subplot(121), plt.imshow(sample[self.gt_key][_i].cpu().numpy().transpose([1, 2, 0]))
                plt.subplot(122), plt.imshow(sample[self.blur_key][_i].cpu().numpy().transpose([1, 2, 0]))
                plt.savefig(os.path.join(self.logger.save_dir, f'{name}_input_sample_{_i}.png'))

        # check for image and kernel shape
        i = 0
        # for i in range(len(self.dataset_paths)):
        #      self.initialize_dataset(self.onthefly, self.dataset_paths[i])
        print(f"gt_key:{self.gt_key} blur_key:{self.blur_key}")
        for sample in self.dataset_train:
            save_images(sample, f'train{i}')
            break
        for sample in self.dataset_val:
            save_images(sample, f'valid{i}')
            break
        for sample in self.dataset_test:
            save_images(sample, f'test_{i}')
            break
        # self.initialize_dataset(self.onthefly, self.dataset_paths[0])

        l = len(self.dataset_train.dataset)
        n = 100
        out = timeit.Timer(lambda: self.dataset_train.dataset.__getitem__(0))
        print(f"Dataset train size {l}: {n} iteration took {np.average(out.repeat(repeat=1, number=n))}")

        l = len(self.dataset_val.dataset)
        out = timeit.Timer(lambda: self.dataset_val.dataset.__getitem__(0))
        print(f"Dataset val size {l}: {n} iteration took {np.average(out.repeat(repeat=1, number=n))}")

        l = len(self.dataset_test.dataset)
        out = timeit.Timer(lambda: self.dataset_test.dataset.__getitem__(0))
        n = 10
        print(f"Dataset test size {l}: {n} iteration took {np.average(out.repeat(repeat=1, number=n))}")

    def setup_dataloaders(self):
        # create train and validation dataloaders
        train_set = create_dataset(self.opt.datasets.train)
        self.dataloader = DataLoader(
            train_set,
            shuffle=self.opt.datasets.train.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 1),
        )
        val_set = create_dataset(self.opt.datasets.val)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 1),
        )

    def prepare(self):
        if not self.trackers_initialized:
            self.prepare_trackers()
            
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.opt.train.gradient_accumulation_steps)
        if self.opt.train.max_train_steps is None:
            self.opt.train.max_train_steps = self.opt.train.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.setup_optimizers()
        self.setup_schedulers()
        for i in range(len(self.models)):
            self.models[i] = self.accelerator.prepare(self.models[i])
        for i in range(len(self.optimizers)):
            self.optimizers[i] = self.accelerator.prepare(self.optimizers[i])
            self.schedulers[i] = self.accelerator.prepare(self.schedulers[i])
        
        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.opt.train.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.opt.train.max_train_steps = self.opt.train.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.opt.train.num_train_epochs = math.ceil(self.opt.train.max_train_steps / num_update_steps_per_epoch)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def setup_optimizers(self):
        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        optim_params = []
        for model in self.models:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print(f"Parameter {k} is not optimized.")

        params_to_optimize = [{'params': optim_params}]
        opt = self.opt.train.optim
        optimizer = optimizer_class(
            params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
        self.optimizers.append(optimizer)

    def setup_schedulers(self):
        for optimizer in self.optimizers:    
            lr_scheduler = get_scheduler(
                self.opt.train.scheduler.type,
                optimizer=optimizer,
                num_warmup_steps=self.opt.train.scheduler.lr_warmup_steps,
                num_training_steps=self.opt.train.max_train_steps,
                num_cycles=self.opt.train.scheduler.lr_num_cycles,
                power=self.opt.train.scheduler.lr_power,
            )
            self.schedulers.append(lr_scheduler)

    def feed_data(self, data, is_train=True):
        pass

    def optimize_parameters(self):
        pass
    
    def save_other_parameters(self, path):
        pass

    def load_other_parameters(self, path):
        pass

    def grids(self, keys, opt):
        """
        Make the input images into grids for large image inference.
        Args:
            keys (list): keys of the input sample.
            opt (dict): options for train/validation.
        """
        self.original_size = {}
        self.minibatch_size = opt.max_minibatch
        for key in keys:
            self.original_size[key] = self.sample[key].size()
            crop_size_h = opt.get('patch_size_h', None)
            crop_size_w = opt.get('patch_size_w', None)
            if crop_size_h is None:
                crop_size_h = int(opt['patch_size_h_ratio'] * self.original_size[key][-2])
            if crop_size_w is None:
                crop_size_w = int(opt['patch_size_w_ratio'] * self.original_size[key][-1])
            overlap = opt['patch_overlap']
            stride_h, stride_w = int(crop_size_h * (1 - overlap)), int(crop_size_w * (1 - overlap))
            patched, patched_pos = patchify(self.sample[key], crop_size_h, crop_size_w, stride_h, stride_w)
            # patched has a shape (... n c h w)

            self.sample[key+'_original'] = self.sample[key]
            self.sample[key+'_patched'] = patched
            self.sample[key+'_patched_pos'] = patched_pos
            self.sample[key] = None  # fill using patched with a minibatch size
    
    # @torch.no_grad()
    def setup_patches(self):
        """
        Yield patches for large image inference.
        """
        keys = list(self.original_size.keys())
        b, n = 0, 0
        for key in keys:
            b = max(b,self.original_size[key][0])
            n = max(n, len(self.sample[key+'_patched_pos']))
        for i in range(b):
            for j in range(0, n, self.minibatch_size):
                for key in keys:
                    _i = i if i < self.original_size[key][0] else 0  # some data always have only 1 batch
                    self.sample[key] = self.sample[key+'_patched'][_i, ..., j:j+self.minibatch_size,:,:,:]
                yield

    def train(self):
        total_batch_size = self.opt.train.batch_size * self.accelerator.num_processes * self.opt.train.gradient_accumulation_steps
        is_patched = self.opt.train.patched
        if is_patched:
            self.logger.info("====== Patched training is enabled =======")
        if not self.is_train:
            self.logger.info("====== Testing only! =======")
            self.opt.train.num_train_epochs = 0
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num batches each epoch = {len(self.dataloader)}")
        self.logger.info(f"  Num Epochs = {self.opt.train.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.opt.train.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.opt.train.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.opt.train.max_train_steps}")
        self.logger.info(f"  Experiments root = {self.opt.path.experiments_root}")

        if self.opt.train.check_speed:
            self.check_model_speed()
        else:
            self.logger.info("Speed check is disabled. Skipping...")

        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.opt.path.resume_from_checkpoint:
            if self.opt.path.resume_from_checkpoint != "latest":
                path = self.opt.path.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.opt.path.resume_from_path)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.opt.path.resume_from_checkpoint}' does not exist in {self.opt.path.resume_from_path}. Starting a new training run."
                )
                self.opt.path.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                load_path = os.path.join(self.opt.path.resume_from_path, path)
                # try:
                self.accelerator.load_state(load_path)
                # except Exception as e:
                #     print(f" {'='*50} Failed to load state {e}")
                self.load_other_parameters(load_path)
                self.global_step = int(path.split("-")[1])

                initial_global_step = self.global_step-1
                first_epoch = self.global_step // self.num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.opt.train.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
        log_freq = np.ceil((self.opt.train.max_train_steps - self.global_step) / 15000)
        
        for epoch in range(first_epoch, self.opt.train.num_train_epochs):
            for step, batch in enumerate(self.dataloader):
                with self.accelerator.accumulate(*self.models):
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                        if self.accelerator.is_main_process:
                            if self.global_step % self.opt.train.checkpointing_steps == 0:
                                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                if self.opt.train.checkpoints_total_limit is not None:
                                    keep_last_checkpoints(self.opt.path.experiments_root, self.opt.train.checkpoints_total_limit, self.logger)

                                save_path = os.path.join(self.opt.path.experiments_root, f"checkpoint-{self.global_step}")
                                self.accelerator.save_state(save_path, save_optimizer=False)
                                self.save_other_parameters(save_path)
                                self.logger.info(f"Saved state to {save_path}")

                            if self.global_step % self.opt.train.validation_steps == 0:
                                self.validation()
                        self.global_step += 1

                    # perform training step
                    self.feed_data(batch, is_train=True)
                    if is_patched:
                        for _ in self.setup_patches():
                            losses = self.optimize_parameters()
                    else:
                        losses = self.optimize_parameters()
                    
                    for i in range(len(self.schedulers)):
                        self.schedulers[i].step()
                        
                    logs = {}
                    for loss_i, loss in losses.items():
                        logs[f"loss_{loss_i}"] = loss.detach().item()
                    for lr_i, lr in enumerate(self.schedulers[0].get_last_lr()):
                        logs[f"lr_{lr_i}"] = lr
                    progress_bar.set_postfix(**logs)

                    if self.global_step % log_freq == 0:
                        self.accelerator.log(logs, step=self.global_step)

                    if self.global_step > self.opt.train.max_train_steps:
                        break

        # END TRAINING
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.validation() # Final validation. If testing script, training loop will be skipped to here
            if self.opt.push_to_hub:
                repo_id = self.opt.hub_model_id or Path(self.opt.experiment_key).name
                # if repo_exists(whoami()['name'] + "/" + repo_id):
                #     self.logger.warning(f"Deleting repo... {repo_id}")
                #     delete_repo(whoami()['name'] + "/" + repo_id)
                repo_id = create_repo(
                    repo_id=repo_id,
                    exist_ok=True,
                    private=True,
                ).repo_id
                # save_model_card(
                #     repo_id,
                #     image_logs=image_logs,
                #     base_model=args.pretrained_model_name_or_path,
                #     repo_folder=args.output_dir,
                # )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=self.opt.path.experiments_root,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*", '*.jpg'],
                )

        self.accelerator.end_training()
        
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()
        
        # dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(5)), 
        #                         shuffle=False, 
        #                         batch_size=1)
        # print(f"Tesing using {len(dataloader)} training data...")
        # dataloader = self.accelerator.prepare(dataloader)
        # for batch in dataloader:
        #     idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        # self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            if idx > self.max_val_steps:
                break

        for model in self.models:
            model.train()
            
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f'{net.__class__.__name__} - '
                           f'{net.module.__class__.__name__}')
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        print(net_str)

    
    