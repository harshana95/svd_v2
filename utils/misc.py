import os
import shutil
import cv2
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch
from os import path as osp
import logging
import os
import diffusers
import torch
import transformers

from skimage.metrics import structural_similarity as ssim
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import wandb
import yaml


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def get_basename(p):
    tmp = os.path.basename(p)
    while tmp == '':
        p = os.path.dirname(p)
        tmp = os.path.basename(p)
    return tmp

def initialize(args, logger, experiment_name):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    args.path.experiments_root = os.path.join(args.path.root, args.name, experiment_name)
    logging_dir = os.path.join(args.path.experiments_root, args.path.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.path.experiments_root,
                                                      logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        # if args.resume_from_checkpoint is None and args.pretrained_model_name_or_path is None:
        #     logger.warning(f"Deleting folder... {args.output_dir}")
        #     try:
        #         shutil.rmtree(args.output_dir)
        #     except FileNotFoundError:
        #         pass
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    return accelerator

def find_attr(modules, attr):
    """Find attribute in module.

    Args:
        modules (file modules): Module.
        attr (str): Attribute name.

    Returns:
        Any: Attribute value.
    """
    for module in modules:
        cls_ = getattr(module, attr, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{attr} is not found.')
    return cls_


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    class DictAsMember(dict):
        def __getattr__(self, name):
            if name not in self.keys():
                # print(f"'{name}' not in {list(self.keys())}")
                return None
            value = self[name]
            return value

    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return DictAsMember(loader.construct_pairs(node))

    Dumper.add_representer(DictAsMember, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def generate_folder(p):
    if not os.path.exists(p):
        os.makedirs(p)


def keep_last_checkpoints(output_dir, checkpoints_total_limit, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)


def log_image(args, accelerator, formatted_images, name, step):
    n, c, h, w = formatted_images.shape
    if c == 1:
        formatted_images = einops.repeat(formatted_images, 'n 1 h w -> n 3 h w')
    if args.val.save_images:
        os.makedirs(os.path.join(args.path.experiments_root, 'images'), exist_ok=True)
        plt.imsave(os.path.join(args.path.experiments_root, 'images' , f"{name}_{step//1000:04d}k.jpg"), einops.rearrange(np.hstack(formatted_images),'c h w -> h w c'))
    name += '.jpg'
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(name, formatted_images, step, dataformats="NCHW")
        elif tracker.name == "wandb":
            tracker.log({"validation": wandb.Image(formatted_images, caption=name)})
        elif tracker.name == "comet_ml":
            tracker.writer.log_image(np.hstack(formatted_images), name=name, step=step, image_channels="first")
        else:
            raise Exception(f"image logging not implemented for {tracker.name}")

def log_metrics(img1, img2, metrics, accelerator, step, comment=""):
    ret = {}
    if len(img1.shape) == 3:
        img1 = einops.rearrange(img1, 'c h w -> h w c')
        img2 = einops.rearrange(img2, 'c h w -> h w c')
    elif len(img1.shape) == 4:
        img1 = einops.rearrange(img1, 'n c h w -> n h w c')
        img2 = einops.rearrange(img2, 'n c h w -> n h w c')
    for metric in metrics:
        if metric == 'psnr':
            out = calculate_psnr(img1, img2, metrics[metric]['crop'])
        elif metric == 'ssim':
            out = calculate_ssim(img1, img2, metrics[metric]['crop'])
        elif metric == 'mse':
            out = calculate_mse(img1, img2, metrics[metric]['crop'])
        elif metric == 'argmax':
            out = np.sqrt(np.mean(np.argmax(img1, axis=-1) - np.argmax(img2, axis=-1))**2)
        else:
            raise Exception(f"Unknown metric {metric}")
        ret[comment+metric]= out
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            raise NotImplementedError()
        elif tracker.name == "wandb":
            raise NotImplementedError()
        elif tracker.name == "comet_ml":
            tracker.writer.log_metrics(ret, step=step)
        else:
            raise Exception(f"image logging not implemented for {tracker.name}")

def calculate_psnr(img1, img2, crop=30):
    mse = calculate_mse(img1*255, img2*255, crop=crop)
    assert mse.min() > 0, f"mse min is {mse.min()}"
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def calculate_mse(img1, img2, crop=30):
    if crop is None:
        return np.mean((img1 - img2) ** 2)
    return np.mean((img1[..., crop:-crop, crop:-crop, :] - img2[..., crop:-crop, crop:-crop, :]) ** 2)

def calculate_ssim(img1, img2, crop=30):
    # Ensure the images are in grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256,256))
    # Compute SSIM between the two images
    if crop is None:
        ssim_value, ssim_map = ssim(img1, img2, full=True, win_size=7)
    else:
        ssim_value, ssim_map = ssim(img1[crop:-crop, crop:-crop], img2[crop:-crop, crop:-crop], full=True, win_size=7)
    return ssim_value, ssim_map

# Convert the figure to a NumPy array
def fig_to_array(fig):
    # Render the figure to a canvas
    fig.canvas.draw()
    # Convert the canvas to a NumPy array (RGB format)
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert from ARGB to RGBA
    data = np.roll(data, 3, axis=2)
    return einops.rearrange(data[...,:3], 'h w c -> c h w')