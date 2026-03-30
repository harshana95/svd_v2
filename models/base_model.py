from datetime import datetime
import gc
import glob
import importlib
import math
import os
from pathlib import Path
import time
import timeit
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from huggingface_hub import create_repo, upload_folder
from safetensors.torch import load_file
import peft

from torch.utils.data import DataLoader
from diffusers import get_scheduler
from tqdm import tqdm
from utils import initialize, keep_last_checkpoints
from utils.dataset_utils import  patchify

from models.archs import find_network_class
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
        save_dir = os.path.join(output_dir, f"{class_name}_{saved[class_name]}")
        os.makedirs(save_dir, exist_ok=True)
        # Prefer HuggingFace / diffusers style saving when available,
        # but fall back to a plain state_dict checkpoint if that fails.
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(save_dir)
            except Exception as e:
                print(f"{'='*50} Failed to save {class_name} with save_pretrained, "
                      f"falling back to torch.save. Error: {e}")
                torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        i -= 1

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        
        ckpt_dir = os.path.join(input_dir, f"{class_name}_{saved[class_name]}")

        # load diffusers style into model
        try:
            try:
                c, _ = find_network_class(class_name)
            except ValueError as e:  # class is not written by us. Try to load from diffusers
                m = importlib.import_module('diffusers') # load the module, will raise ImportError if module cannot be loaded
                c = getattr(m, class_name)  # get the class, will raise AttributeError if class cannot be found    
            
            assert c is not None
            if hasattr(c, 'config_class'):
                assert c.config_class is not None
            else:
                print(f"********* Class {class_name} does not have a config_class") 

            if "PeftModel" in class_name:
                model = peft.PeftModel.from_pretrained(model.base_model.model, ckpt_dir, is_trainable=True)
            else:
                combined_state_dict = {}
                for file_path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
                    state_dict_part = load_file(file_path)
                    combined_state_dict.update(state_dict_part)
                model.load_state_dict(combined_state_dict)
                
        except Exception as e:
            # Fallback: load plain torch checkpoint produced by save_model_hook
            try:
                bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
                state = torch.load(bin_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                # Handle the common case where a submodule state_dict was saved
                # (e.g. keys like "0.weight" instead of "net.0.weight").
                if isinstance(state, dict) and state and not any(k.startswith("net.") for k in state.keys()):
                    if hasattr(model, "net") and all(k.split(".", 1)[0].isdigit() for k in state.keys()):
                        state = {f"net.{k}": v for k, v in state.items()}
                missing, unexpected = model.load_state_dict(state, strict=False)
                if missing:
                    print(f"[{class_name}] Missing keys: {missing}")
                if unexpected:
                    print(f"[{class_name}] Unexpected keys: {unexpected}")
            except Exception as e2:
                print(f"{'='*50} Failed to load {class_name} {'='*50} {e} {c} {model}")
                print(f"{'='*50} Torch fallback also failed for {class_name} {'='*50} {e2}")

class BaseModel():
    """Base model."""
    FLOPS_SAVED = False

    def __init__(self, opt, logger=None):
        self.is_train = opt['is_train']
        if not self.is_train:
            opt['name'] = 'infer'
            opt['tracker_project_name'] = 'infer'
        self.experiment_name = opt.experiment_key + f"{opt.comment}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        self.accelerator = initialize(opt, logger, self.experiment_name)
        logger.info(opt)
        self.opt = opt
        self.logger = logger
        
        self.save_handle = self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.schedulers = []
        self.optimizers = []
        self.models = []
        self.overrode_max_train_steps = False
        self.global_step = 0
        self.max_val_steps = opt.val.get('max_val_steps', 10)
        if not self.is_train:
            self.max_val_steps = 1e6

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
        
        if self.is_train:
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
                    tracker.writer.log_code(
                        folder=".", 
                        pattern="*.py", 
                        folder_name='source-code',
                        overwrite=True)
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
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 0),
        )
        val_set = create_dataset(self.opt.datasets.val)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=self.opt.datasets.val.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 0),
        )

    def prepare(self):
        if not self.trackers_initialized:
            self.prepare_trackers()
        if self.is_train:
            # Scheduler and math around the number of training steps.
            num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.opt.train.gradient_accumulation_steps)
            if self.opt.train.max_train_steps is None:
                self.opt.train.max_train_steps = self.opt.train.num_train_epochs * num_update_steps_per_epoch
                self.overrode_max_train_steps = True

            self.setup_optimizers()
            self.setup_schedulers()

        # prepare models, datasets, and optimizers
        for i in range(len(self.models)):
            self.models[i] = self.accelerator.prepare(self.models[i])
        for i in range(len(self.optimizers)):
            self.optimizers[i] = self.accelerator.prepare(self.optimizers[i])
            self.schedulers[i] = self.accelerator.prepare(self.schedulers[i])
        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        self.print_model_memory()

        if self.is_train:
            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.opt.train.gradient_accumulation_steps)
            if self.overrode_max_train_steps:
                self.opt.train.max_train_steps = self.opt.train.num_train_epochs * num_update_steps_per_epoch
            # Afterwards we recalculate our number of training epochs
            self.opt.train.num_train_epochs = math.ceil(self.opt.train.max_train_steps / num_update_steps_per_epoch)
            self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def _sanitize_optimizer_state_shapes(self):
        """Drop stale optimizer states whose tensor shapes no longer match params.

        This can happen when resuming from checkpoints after architecture/config
        changes (e.g., channel count, LoRA ranks, adapter settings).
        """
        for optimizer in self.optimizers:
            # Iterate param groups and remove only incompatible state entries.
            for group in optimizer.param_groups:
                for param in group["params"]:
                    state = optimizer.state.get(param, None)
                    if not state:
                        continue

                    invalid_state = False
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        buf = state.get(key, None)
                        if torch.is_tensor(buf) and buf.shape != param.shape:
                            invalid_state = True
                            break

                    if invalid_state:
                        optimizer.state.pop(param, None)

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

    def forwardpass(self, *args, **kwargs):
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
        # for i in range(b):
        #     for j in range(0, n, self.minibatch_size):
        #         for key in keys:
        #             _i = i if i < self.original_size[key][0] else 0  # some data always have only 1 batch
        #             self.sample[key] = self.sample[key+'_patched'][_i, ..., j:j+self.minibatch_size,:,:,:]
        #         yield
        
        
        for j in range(0, n, self.minibatch_size):
            for key in keys:
                self.sample[key] = self.sample[key+'_patched'][:, ..., j:j+self.minibatch_size,:,:,:]
                self.sample[key] = einops.rearrange(self.sample[key], "b ... mb c h w -> (b mb) ... c h w")
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
                try:
                    self.accelerator.load_state(load_path)
                except Exception as e:
                    self.accelerator.print(f" {'='*50} Failed to load state {e}")
                # Optimizer moments can be stale when model param shapes changed.
                # Remove incompatible entries so Adam re-initializes them lazily.
                self._sanitize_optimizer_state_shapes()
                self.load_other_parameters(load_path)
                self.global_step = int(path.split("-")[1])

                initial_global_step = self.global_step-1
                if self.is_train:
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
                            if self.global_step % self.opt.train.checkpointing_steps == 0 and self.global_step > 1:
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
            
    def print_model_memory(self):
        """Print the memory usage of the models."""
        # print number of trainable parameters
        total_memory_usage = 0
        total_trainable_params = 0
        total_params = 0
        for m in self.models:
            all_param = 0
            trainable_params = 0
            memory_usage = 0
            for name, param in m.named_parameters():
                all_param += param.numel()
                memory_usage += param.numel() * param.element_size() # bytes
                if param.requires_grad:
                    trainable_params += param.numel()
            print(f"{m.__class__.__name__} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param} || memory: {memory_usage / (1024 ** 2):.2f} MB")
            total_memory_usage += memory_usage
            total_trainable_params += trainable_params
            total_params += all_param
        self.logger.info(f"Total memory usage: {total_memory_usage / (1024 ** 2):.2f} MB || Total trainable params: {total_trainable_params} || Total params: {total_params}")

    @torch.no_grad()
    def calculate_flops(self, *args, n=10, **kwargs):
        """
        Estimate FLOPs for a image input.

        Args:
            *args: arguments to the forward pass
        """
        if self.FLOPS_SAVED:
            return

        # calculate flops
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            self.logger.warning("torch.utils.flop_counter is not installed; cannot compute FLOPs.")
            return

        flop_counter = FlopCounterMode(display=False, depth=None)
        
        with flop_counter:
            self.forwardpass(*args)
            
        total_flops = flop_counter.get_total_flops()
        table = flop_counter.get_table()

        # calculate time taken to forward pass
        time_list = []
        for _ in range(n):
            start_time = time.time()
            self.forwardpass(*args)
            end_time = time.time()
            time_list.append(end_time - start_time)
        
        # save table to file in experiments root
        with open(os.path.join(self.opt.path.experiments_root, self.opt.path.logging_dir, "flops_table.txt"), "w") as f:
            f.write(table)
        with open(os.path.join(self.opt.path.experiments_root, self.opt.path.logging_dir, "flops_summary.txt"), "w") as f:
            f.writelines([f"Total FLOPs: {total_flops / 1e9:.3f} GFLOPs\n",
                            f"Average time taken to forward pass: {np.mean(time_list)} seconds\n",
                            f"std: {np.std(time_list)} seconds\n",
                            f"min: {np.min(time_list)} seconds\n",
                            f"max: {np.max(time_list)} seconds\n"])
        
        self.FLOPS_SAVED = True

        self.logger.info(
                f"Estimated FLOPs: {total_flops / 1e9:.3f} GFLOPs || Average time taken to forward pass: {np.mean(time_list)} seconds || std: {np.std(time_list)} seconds || min: {np.min(time_list)} seconds || max: {np.max(time_list)} seconds"
            )
