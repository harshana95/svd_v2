from datetime import datetime
import gc
import glob
import importlib
import math
import os
from pathlib import Path
import shutil
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

from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import LambdaLR
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

        # A module that simply does not exist in this checkpoint (e.g. an
        # adapter added after the checkpoint was written) must be SKIPPED and
        # left at its initialization -- not treated as an error. Before this
        # guard the two load attempts below both failed and the handler's own
        # message referenced an unbound `c`, raising UnboundLocalError OUT of
        # load_model_hook and aborting the whole loop, so EVERY REMAINING
        # MODULE silently went unloaded while the run looked healthy.
        if not os.path.isdir(ckpt_dir):
            # A connector-family module missing while a DIFFERENT connector-family folder exists is a config/checkpoint
            # mismatch (e.g. an eval yml without `vjepa_layers` loading a multi-layer checkpoint): leaving the connector at
            # random init would silently produce garbage, so fail loudly. Genuinely new modules are still skipped.
            _family = ("MultiLayerDynamicsConnector", "JEPAFactoredConnector", "JEPAPerceptualEncoder")
            if class_name in _family:
                _present = [d for d in os.listdir(input_dir) if any(d.startswith(f + "_") for f in _family)]
                if _present:
                    raise RuntimeError(
                        f"Connector mismatch: model has {class_name}_{saved[class_name]} but {input_dir} contains {_present}. "
                        f"Fix the config (vjepa_layers / dynamics_connector_type) to match the checkpoint.")
            print(f"[{class_name}_{saved[class_name]}] not in checkpoint -- leaving at init (new module)")
            continue

        # load diffusers style into model
        c = None  # referenced by the failure message below; must always be bound
        try:
            if "PeftModel" in class_name:
                # peft's PeftModel isn't a diffusers/local arch class, so it
                # can't be resolved via find_network_class or diffusers'
                # namespace (getattr(diffusers, "PeftModel") raises
                # AttributeError, not the ValueError the branch below
                # expects) — handle it before that lookup runs at all.
                #
                # Must load in place via set_peft_model_state_dict, NOT
                # `peft.PeftModel.from_pretrained(model.base_model.model, ...)`
                # — that call was confirmed (empirically, via
                # `id(param_before) != id(param_after)`) to construct brand
                # new nn.Parameter objects for every LoRA weight, silently
                # replacing the ones already injected into the shared base
                # transformer. Any optimizer built before this resume (i.e.
                # every optimizer in this codebase, since setup_optimizers()
                # always runs during prepare(), before train() ever resumes a
                # checkpoint) still holds `param_groups[...]['params']`
                # referencing the OLD, now-orphaned Parameter objects: the
                # forward/backward pass runs on the NEW ones (so losses and
                # saved checkpoints look completely normal), but
                # optimizer.step() reads .grad from the stale objects, which
                # is always None post-resume, so AdamW silently no-ops on
                # every single call for the rest of that process's lifetime.
                # Confirmed via direct weight-file checksums across three
                # independent resumed training lines in this codebase
                # (Flux2KleinDistill, Flux2KleinDistillVJEPA,
                # Flux2KleinHybrid): every "continued" checkpoint was
                # byte-identical to the checkpoint it resumed from, despite
                # 20,000+ further training steps and varying hyperparameters
                # (including learning rate) — see the
                # peft_resume_optimizer_disconnect memory note.
                # set_peft_model_state_dict updates the *existing* Parameter
                # objects' data in place (verified: `id(param) unchanged`,
                # values correctly overwritten), keeping every already-built
                # optimizer's references valid.
                from peft.utils import set_peft_model_state_dict
                combined_state_dict = {}
                for file_path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
                    combined_state_dict.update(load_file(file_path))
                set_peft_model_state_dict(model, combined_state_dict, adapter_name="default")
            else:
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
                # Drop tensors whose SHAPE disagrees with the live module, then
                # load the rest. strict=False tolerates missing/unexpected keys
                # but still RAISES on a size mismatch, which would abandon the
                # whole module over one tensor -- e.g. resuming a cross-attention
                # stack across a tubelet-count change mismatches only `pos_t`
                # (28,800 params, 0.11% of the module) yet would otherwise
                # discard all 25M trained weights and silently restart it from
                # zero-init.
                own = model.state_dict()
                mismatched = [k for k, v in state.items()
                              if k in own and tuple(own[k].shape) != tuple(v.shape)]
                # PARTIAL COPY rather than discard. A resized table (typically a
                # positional embedding whose length tracks a count that changed
                # -- e.g. WanDynamicsCrossAttn.pos_t when the max burst size
                # goes 3 -> 5, giving (12,64) -> (20,64)) shares its leading
                # rows with the checkpoint. Dropping it entirely restarts a
                # LEARNED tensor from scratch: for pos_t that discards the
                # cross-attention's whole notion of which tubelet is which,
                # which is the one thing keeping the KV from being an unordered
                # bag. Copying the overlapping region keeps every position the
                # checkpoint actually trained and leaves only genuinely-new
                # entries at init. Works in both directions (grow or shrink).
                # Only applied when the rank matches -- anything else is a real
                # architecture change, not a resize, and is still skipped.
                dropped = []
                for k in mismatched:
                    ck, mk = state[k], own[k]
                    if ck.dim() != mk.dim():
                        dropped.append(k)
                        continue
                    sl = tuple(slice(0, min(a, b)) for a, b in zip(ck.shape, mk.shape))
                    merged = mk.clone()
                    merged[sl] = ck[sl].to(merged.dtype)
                    state[k] = merged
                    kept = "x".join(str(x.stop) for x in sl)
                    print(f"[{class_name}] RESIZED: {k} checkpoint{tuple(ck.shape)} -> "
                          f"model{tuple(mk.shape)}; copied the overlapping {kept} block, "
                          f"remainder left at initialization")
                if dropped:
                    for k in dropped:
                        print(f"[{class_name}] SHAPE MISMATCH, not loaded: {k} "
                              f"checkpoint{tuple(state[k].shape)} vs model{tuple(own[k].shape)} "
                              f"-- differing rank, this tensor stays at its initialization")
                    state = {k: v for k, v in state.items() if k not in dropped}
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
        use_same_dir = bool(opt.path.get('use_same_dir', False))
        resume_from_checkpoint = opt.path.get('resume_from_checkpoint')
        resume_from_path = opt.path.get('resume_from_path')
        if use_same_dir and resume_from_checkpoint and resume_from_path:
            self.experiment_name = os.path.basename(os.path.normpath(resume_from_path))
            self.accelerator = initialize(opt, logger, self.experiment_name, experiments_root=resume_from_path)
        else:
            self.experiment_name = opt.experiment_key + f"{opt.comment if opt.comment else ''}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
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
        # Under -infer (is_train=False), default to sweeping the whole split (1e6) rather
        # than the training-time quick-check default (10) -- but an explicit max_val_steps
        # in the config still wins either way, so an eval config can bound the image count.
        default_max_val_steps = 1e6 if not self.is_train else 10
        self.max_val_steps = opt.val.get('max_val_steps', default_max_val_steps)

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
            self.logger.info(f"Initializing trackers with tracker_kwargs: {tracker_kwargs}")
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
        # A dataset can be an IterableDataset that yields already-batched
        # tensors itself (e.g. FullResPatchifierDataset) -- DataLoader must
        # use batch_size=None in that case, or it would try to re-batch
        # already-batched groups.
        self.logger.info("Setting up val dataloaders...")
        val_set = create_dataset(self.opt.datasets.val)
        val_iterable = isinstance(val_set, IterableDataset)
        self.test_dataloader = DataLoader(
            val_set,
            shuffle=False if val_iterable else self.opt.datasets.val.use_shuffle,
            batch_size=None if val_iterable else self.opt.val.batch_size,
            num_workers=self.opt.datasets.val.get('num_worker_per_gpu', 0),
            # A dataset may optionally define its own collate_fn (e.g.
            # Adobe240Dataset's batch-consistent variable-aspect-ratio crop,
            # which needs to see a whole batch at once) -- None for any
            # dataset that doesn't, which is DataLoader's own default.
            collate_fn=None if val_iterable else getattr(val_set, "collate_fn", None),
        )

        self.logger.info("Setting up train dataloader...")
        train_set = create_dataset(self.opt.datasets.train)
        train_iterable = isinstance(train_set, IterableDataset)
        self.dataloader = DataLoader(
            train_set,
            shuffle=False if train_iterable else self.opt.datasets.train.use_shuffle,
            batch_size=None if train_iterable else self.opt.train.batch_size,
            num_workers=self.opt.datasets.train.get('num_worker_per_gpu', 0),
            collate_fn=None if train_iterable else getattr(train_set, "collate_fn", None),
        )

    def prepare(self):
        self.logger.info("Preparing model...")
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
            self.logger.info("Optimizers and schedulers prepared...")

        # prepare models, datasets, and optimizers
        for i in range(len(self.models)):
            self.models[i] = self.accelerator.prepare(self.models[i])
        for i in range(len(self.optimizers)):
            self.optimizers[i] = self.accelerator.prepare(self.optimizers[i])
            self.schedulers[i] = self.accelerator.prepare(self.schedulers[i])
        self.dataloader = self.accelerator.prepare(self.dataloader)
        self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        # Optional secondary validation dataloader (see e.g.
        # VAEBlurAlignFluxLoRA_DP_model.py's `datasets.val_secondary`) --
        # only a few model files set this attribute; every other subclass
        # never has it, hence the getattr guard rather than a plain
        # self.test_dataloader_secondary reference.
        if getattr(self, "test_dataloader_secondary", None) is not None:
            self.test_dataloader_secondary = self.accelerator.prepare(self.test_dataloader_secondary)
        self.logger.info("Models, datasets, and optimizers prepared...")

        self.print_model_memory()

        if self.is_train:
            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(len(self.dataloader) / self.opt.train.gradient_accumulation_steps)
            if self.overrode_max_train_steps:
                self.opt.train.max_train_steps = self.opt.train.num_train_epochs * num_update_steps_per_epoch
            # Afterwards we recalculate our number of training epochs
            self.opt.train.num_train_epochs = math.ceil(self.opt.train.max_train_steps / num_update_steps_per_epoch)
            self.num_update_steps_per_epoch = num_update_steps_per_epoch

    def snapshot_python_sources(self):
        """Save a checkpoint copy of repository Python files in the logging directory."""
        if not self.accelerator.is_main_process:
            return

        project_root = Path(__file__).resolve().parent.parent
        logging_root = Path(self.opt.path.experiments_root) / self.opt.path.logging_dir
        snapshot_dir = logging_root / f"python_source_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        skipped_dirs = {".git", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".pytest_cache", ".idea", ".cursor"}
        copied_count = 0

        for py_file in project_root.rglob("*.py"):
            relative_parts = set(py_file.relative_to(project_root).parts)
            if relative_parts.intersection(skipped_dirs):
                continue
            try:
                rel_path = py_file.relative_to(project_root)
                dest_file = snapshot_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(py_file, dest_file)
                copied_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to snapshot file {py_file}: {e}")

        self.logger.info(f"Saved Python source snapshot to {snapshot_dir} ({copied_count} files)")

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
        # "cosine_floor": like diffusers' plain "cosine" (same
        # num_warmup_steps/num_training_steps semantics -- including the
        # "set lr_warmup_steps to the absolute resume step, not 0" rule for
        # resuming into a schedule change, see
        # findings/Wan22VJEPADynamics_findings.md / the accelerate_save_
        # optimizer_noop memory) but decays toward a FLOOR (train.scheduler.
        # lr_floor, absolute LR, default 0.0) instead of exactly 0 -- not
        # expressible via diffusers.get_scheduler, which has no floor
        # parameter for any built-in schedule. Built directly as a LambdaLR
        # multiplier (matching every HF schedule's own convention: the
        # lambda multiplies optimizer.param_groups[i]['lr'] at construction
        # time), so this is a plain addition alongside get_scheduler's
        # existing types -- everything else (constant, constant_with_warmup,
        # linear, cosine, polynomial, ...) is unchanged.
        for optimizer in self.optimizers:
            sched_type = self.opt.train.scheduler.type
            if sched_type == "cosine_floor":
                base_lr = float(self.opt.train.optim.learning_rate)
                floor_lr = float(self.opt.train.scheduler.get("lr_floor", 0.0))
                floor_ratio = floor_lr / base_lr if base_lr > 0 else 0.0
                warmup = int(self.opt.train.scheduler.lr_warmup_steps)
                total = int(self.opt.train.max_train_steps)

                def lr_lambda(current_step, _warmup=warmup, _total=total, _floor=floor_ratio):
                    if current_step < _warmup:
                        return float(current_step) / float(max(1, _warmup))
                    progress = float(current_step - _warmup) / float(max(1, _total - _warmup))
                    progress = min(max(progress, 0.0), 1.0)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return _floor + (1.0 - _floor) * cosine

                lr_scheduler = LambdaLR(optimizer, lr_lambda)
            else:
                lr_scheduler = get_scheduler(
                    sched_type,
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

    def after_resume_checkpoint(self, load_path):
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


    def save_load_roundtrip_check(self, save_path=None, perturb: float = 1.0):
        """Prove that EVERY trainable parameter actually survives a
        save -> load round trip, before spending a run on it.

        Why this exists: the failure mode in this codebase is silent. A module
        whose class name changed, or that the save hook never visited, simply
        does not appear in the checkpoint; `load_state_dict(strict=False)`
        reports nothing and the module keeps its random init. Nothing errors,
        the run trains, and the defect only surfaces as bad numbers days later.

        The check is only meaningful because it PERTURBS the weights between
        the save and the load. A plain save->load->compare would pass even if
        loading were a complete no-op, which is precisely the bug being hunted.

        Steps: snapshot trainable params -> save_state -> add noise to every
        trainable param -> load_state -> compare against the snapshot. Any
        parameter that does not come back EXACTLY was not really persisted.

        Returns (ok, report). Leaves the checkpoint on disk so it can be fed to
        a downstream stage (e.g. Stage C consuming Stage A/B checkpoint-0).
        """
        import shutil
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return True, "skipped (non-main process)"
        save_path = save_path or os.path.join(self.opt.path.experiments_root, "checkpoint-0")

        # 1. snapshot
        snap = {}
        for mi, model in enumerate(self.models):
            tag = f"{type(model).__name__}_{mi}"
            for n, prm in model.named_parameters():
                if prm.requires_grad:
                    snap[f"{tag}::{n}"] = prm.detach().clone().float().cpu()
        if not snap:
            return True, "no trainable parameters to check"
        self.logger.info("save/load round-trip check: %d trainable tensors across %d modules",
                         len(snap), len(self.models))

        # 2. save
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        self.accelerator.save_state(save_path, save_optimizer=False)
        self.save_other_parameters(save_path)

        # 3. perturb -- without this the check is vacuous
        with torch.no_grad():
            for model in self.models:
                for prm in model.parameters():
                    if prm.requires_grad:
                        prm.add_(torch.randn_like(prm) * perturb)

        # 4. load back
        self.accelerator.load_state(save_path)
        self.load_other_parameters(save_path)

        # 5. compare
        bad, worst = [], {}
        for mi, model in enumerate(self.models):
            tag = f"{type(model).__name__}_{mi}"
            for n, prm in model.named_parameters():
                if not prm.requires_grad:
                    continue
                key = f"{tag}::{n}"
                ref = snap.get(key)
                if ref is None:
                    continue
                d = (prm.detach().float().cpu() - ref).abs().max().item()
                worst[tag] = max(worst.get(tag, 0.0), d)
                if d > 1e-5:
                    bad.append((key, d))

        lines = [f"  {t:<34} max|delta| = {d:.3e}  {'OK' if d <= 1e-5 else 'FAILED'}"
                 for t, d in sorted(worst.items())]
        report = "\n".join(lines)
        if bad:
            report += f"\n  {len(bad)} tensor(s) did NOT round-trip; first few:\n"
            report += "\n".join(f"    {k}  delta={d:.3e}" for k, d in bad[:10])
        self.accelerator.wait_for_everyone()
        return (not bad), report

    def train(self):
        total_batch_size = self.opt.train.batch_size * self.accelerator.num_processes * self.opt.train.gradient_accumulation_steps
        is_patched = self.opt.train.patched
        self.snapshot_python_sources()
        
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
                _skip_opt = bool(self.opt.path.get("resume_skip_optimizer", False))
                _saved = (self.accelerator._optimizers, self.accelerator._schedulers)
                try:
                    if _skip_opt:
                        # warm start into a DIFFERENT param-group layout (e.g. extra
                        # trainable predictor LoRA): restore weights only, fresh optimizer/scheduler
                        self.accelerator._optimizers, self.accelerator._schedulers = [], []
                        self.accelerator.print("resume_skip_optimizer: loading weights only")
                    self.accelerator.load_state(load_path)
                except Exception as e:
                    # HARD FAIL. Continuing here means running with whatever the
                    # modules were initialised to -- and because the adapters are
                    # ZERO-init by design (cross-attention to_out, LoRA B), the
                    # model silently degrades to base Wan and still emits
                    # plausible-looking video. On 2026-09-23 a whole burst-size
                    # sweep was produced this way: its checkpoint had been
                    # rotated away by checkpoints_total_limit while the run that
                    # owned it kept training, the error was printed, and three
                    # eval arms of garbage were scored before the artifacts were
                    # spotted by eye. An unloadable checkpoint is never
                    # recoverable-by-continuing.
                    raise RuntimeError(
                        f"FAILED TO LOAD CHECKPOINT {load_path}: {e}\n"
                        "Refusing to continue -- the model would run with "
                        "initialisation-value adapters and produce output that "
                        "looks trained but is not. If the checkpoint was rotated "
                        "away by checkpoints_total_limit, copy it somewhere "
                        "outside the training run's directory first."
                    ) from e
                self.accelerator._optimizers, self.accelerator._schedulers = _saved
                # Optimizer moments can be stale when model param shapes changed.
                # Remove incompatible entries so Adam re-initializes them lazily.
                self._sanitize_optimizer_state_shapes()
                self.load_other_parameters(load_path)
                self.after_resume_checkpoint(load_path)
                self.global_step = int(path.split("-")[1])

                initial_global_step = self.global_step-1
                if self.is_train:
                    first_epoch = self.global_step // self.num_update_steps_per_epoch
        else:
            initial_global_step = 0

        # Prove the checkpoint round trip BEFORE training. Runs after resume so
        # it validates the state actually about to be trained, and leaves a
        # checkpoint-0 on disk that a downstream stage can consume.
        if self.is_train and bool(self.opt.train.get("save_load_check", True)):
            ok, report = self.save_load_roundtrip_check()
            self.logger.info("===== SAVE/LOAD ROUND-TRIP CHECK =====\n%s", report)
            if not ok:
                raise RuntimeError(
                    "save/load round-trip check FAILED -- at least one trainable tensor "
                    "did not survive save->load, so this run would train weights that are "
                    "silently discarded at the next resume. See the report above. "
                    "Set train.save_load_check: false to bypass (not recommended)."
                )
            self.logger.info("save/load round-trip check PASSED; checkpoint-0 written to %s",
                             os.path.join(self.opt.path.experiments_root, "checkpoint-0"))

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
            # SAVE THE FINAL STATE. The in-loop save fires at the START of an
            # iteration using the PRE-increment global_step (checkpoint-N means
            # "N steps completed"), so the state reached by the LAST iteration is
            # never checked against a boundary: the loop exits with
            # global_step == the completed step count and nothing writes it. On
            # 2026-09-24 a 58000->70000 resume ended by epoch exhaustion at
            # global_step 70000 and the newest checkpoint was 69000 -- ~1000
            # steps of a finished run were lost. (Resuming from 58000 gives
            # first_epoch=29 and num_train_epochs=35, i.e. exactly 12000
            # iterations, the last of which checks global_step=69999.)
            #
            # Placed BEFORE the final validation so the weights are durable even
            # if that (long, at large burst sizes) validation crashes.
            # Skipped when not training (eval/-infer runs also reach this point
            # with the loop skipped) and when global_step is <= 1 or the
            # checkpoint already exists (max_train_steps landing exactly on a
            # checkpointing boundary).
            if self.is_train and self.global_step > 1:
                final_path = os.path.join(self.opt.path.experiments_root, f"checkpoint-{self.global_step}")
                if os.path.isdir(final_path):
                    self.logger.info(f"Final state already saved at {final_path}")
                else:
                    if self.opt.train.checkpoints_total_limit is not None:
                        keep_last_checkpoints(self.opt.path.experiments_root, self.opt.train.checkpoints_total_limit, self.logger)
                    self.accelerator.save_state(final_path, save_optimizer=False)
                    self.save_other_parameters(final_path)
                    self.logger.info(f"Saved FINAL state to {final_path}")
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
