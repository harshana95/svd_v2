import cv2
import einops
import numpy as np
import torch
import os
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
from PIL import Image

from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import (
    cast_training_params,
    free_memory,
)
from diffusers.utils import export_to_video

from models.MotionBlur2Video.helpers import random_insert_latent_frame, transform_intervals
from models.archs import define_network
from models.archs.related.Blur2Vid.cogvideo_arch import CogVideoXTransformer3D_arch
from models.base_model import BaseModel
from models.archs.related.LD.loss import GANLoss
from pipelines.ControlnetCogVideoXPipeline import ControlnetCogVideoXPipeline
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics
from utils.misc import log_video

def encode_video(video, accelerator, vae):
    video = video.to(accelerator.device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds

def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
):
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    free_memory() #delete the pipeline to free up memory

    return videos


class Blur2Vid_model(BaseModel):
    """Generate video from motion blurred image"""

    def __init__(self, opt, logger):
        super(Blur2Vid_model, self).__init__(opt, logger)
        args = self.opt['args']

        # Prepare models and scheduler
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )

        # # CogVideoX-2b weights are stored in float16
        # config = CogVideoXTransformer3D_arch.load_config(
        # os.path.join(args.base_dir, args.pretrained_model_name_or_path),
        # subfolder="transformer",
        # revision=args.revision,
        # variant=args.variant,
        # )

        load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
        transformer = CogVideoXTransformer3D_arch.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            low_cpu_mem_usage=False,
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

        scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

        # We only train the additional adapter controlnet layers
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        
        self.vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        self.model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer

        if self.is_train:
            self.transformer.requires_grad_(True)
            self.init_training_settings()

        self.models.append(self.transformer)

    def init_training_settings(self):
        self.transformer.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # Generator optim params
        optim_params = []
        for k, v in self.transformer.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            
        self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(self.optimizer_g)

        # # Discriminator optim params
        # d_optim_params = []
        # for k, v in self.discriminator.named_parameters():
        #     if v.requires_grad:
        #         d_optim_params.append(v)
        #     else:
        #         self.logger.warning(f'Params {k} will not be optimized.')
        # self.optimizer_d = torch.optim.Adam([{'params': d_optim_params}],
        #     lr=train_opt.optim.learning_rate,
        #     betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
        #     weight_decay=train_opt.optim.adam_weight_decay,
        #     eps=train_opt.optim.adam_epsilon,
        # )
        # self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        model_input = encode_video(self.sample["videos"], self.accelerator, self.vae).to(dtype=self.weight_dtype)  # [B, F, C, H, W]
        prompts = self.sample["prompts"]
        image_latent = encode_video(self.sample["blur_img"], self.accelerator, self.vae).to(dtype=self.weight_dtype)  # [B, F, C, H, W]
        input_intervals = self.sample["input_intervals"]
        output_intervals = self.sample["output_intervals"] 

        batch_size = len(prompts)
        # True = use real prompt (conditional); False = drop to empty (unconditional)
        guidance_mask = torch.rand(batch_size, device=self.accelerator.device) >= 0.2  

        # build a new prompts list: keep the original where mask True, else blank
        per_sample_prompts = [
            prompts[i] if guidance_mask[i] else ""
            for i in range(batch_size)
        ]
        prompts = per_sample_prompts

        # encode prompts
        prompt_embeds = compute_prompt_embeddings(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self.model_config.max_text_seq_length,
            self.accelerator.device,
            self.weight_dtype,
            requires_grad=False,
        )

        # Sample noise that will be added to the latents
        noise = torch.randn_like(model_input)
        batch_size, num_frames, num_channels, height, width = model_input.shape

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
        )
        timesteps = timesteps.long()

        # Prepare rotary embeds
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=self.args.height,
                width=self.args.width,
                num_frames=num_frames,
                vae_scale_factor_spatial=self.vae_scale_factor_spatial,
                patch_size=self.model_config.patch_size,
                attention_head_dim=self.model_config.attention_head_dim,
                device=self.accelerator.device,
            )
            if self.model_config.use_rotary_positional_embeddings
            else None
        )

        # Add noise to the model input according to the noise magnitude at each timestep (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

        input_intervals = transform_intervals(input_intervals, frames_per_latent=4)
        output_intervals = transform_intervals(output_intervals, frames_per_latent=4)
        
        #first interval is always rep
        noisy_model_input, target, condition_mask, intervals = random_insert_latent_frame(image_latent, noisy_model_input, model_input, input_intervals, output_intervals, special_info=self.args.special_info)
        
        for i in range(batch_size):
            if not guidance_mask[i]:
                noisy_model_input[i][condition_mask[i]] = 0

        # Predict the noise residual
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            intervals=intervals,
            condition_mask=condition_mask,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        #this line below is also scaling the input which is bad - so the model is also learning to scale this input latent somehow
        #thus, we need to replace the first frame with the original frame later
        model_pred = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (model_pred[~condition_mask] - target[~condition_mask]) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        self.accelerator.backward(loss)

        for i in range(len(self.optimizers)):
            self.optimizers[i].step()
        return {
            'all': loss
        }
    
    @torch.no_grad()
    def validation(self):
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()

        self.pipeline = ControlnetCogVideoXPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path,
            transformer=self.accelerator.unwrap_model(self.transformer),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            vae=self.accelerator.unwrap_model(self.vae),
            scheduler=self.scheduler,
            torch_dtype=self.weight_dtype,
        )
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx)
            if idx >= self.max_val_steps:
                break
        for model in self.models:
            model.train()

    def validate_step(self, _batch, idx):
        self.feed_data(_batch, is_train=False)
        output_dir = os.path.join(self.opt.path.experiments_root, 'outputs')

        frame = ((self.sample["blur_img"][0].permute(0,2,3,1).cpu().numpy() + 1)*127.5).astype(np.uint8)
        pipeline_args = {
                            "prompt": "",
                            "negative_prompt": "",
                            "image": frame,
                            "input_intervals": self.sample["input_intervals"][0:1],
                            "output_intervals": self.sample["output_intervals"][0:1],
                            "guidance_scale": self.args.guidance_scale,
                            "use_dynamic_cfg": self.args.use_dynamic_cfg,
                            "height": self.args.height,
                            "width": self.args.width,
                            "num_frames": self.args.max_num_frames,
                            "num_inference_steps": self.args.num_inference_steps,
                        }
        assert len(self.sample['file_names']) == 1
        filename = self.sample['file_names'][0]
        name = os.path.basename(os.path.splitext(filename)[0])
        modified_filename = os.path.splitext(filename)[0] + ".mp4"
        gt_file_name = os.path.join(output_dir, "gt", modified_filename)

        num_frames = self.sample["num_frames"][0]        

        #save the gt_video output
        gt_video = self.sample["videos"][0].permute(0,2,3,1).cpu().numpy()
        gt_video = ((gt_video + 1) * 127.5)/255
        gt_video = gt_video[0:num_frames]
        save_frames_as_pngs((gt_video*255).astype(np.uint8), gt_file_name.replace(".mp4", "").replace("gt", "gt_frames"))
        log_video(self.opt, self.accelerator, gt_video, f'gt_{name}', self.global_step)

        #save the blurred image
        log_image(self.opt, self.accelerator, np.clip(frame[0], 0,1), f'blur_{name}', self.global_step)

        videos = log_validation(
            pipe=self.pipeline,
            args=self.args,
            accelerator=self.accelerator,
            pipeline_args=pipeline_args
        )

        #save the output video frames as pngs (uncompressed results) and mp4 (compressed results easily viewable)
        video = videos[0][0:num_frames]
        save_frames_as_pngs((video*255).astype(np.uint8), gt_file_name.replace(".mp4", "").replace("gt", "deblur_frames"))
        log_video(self.opt, self.accelerator, video, f'deblur_{name}', self.global_step)
        
        self.accelerator.wait_for_everyone()
        return idx+1
    
    
def save_frames_as_pngs(video_array,output_dir, 
                        downsample_spatial=1,   # e.g. 2 to halve width & height
                        downsample_temporal=1): # e.g. 2 to keep every 2nd frame
    """
    Save each frame of a (T, H, W, C) numpy array as a PNG with no compression.
    """
    assert video_array.ndim == 4 and video_array.shape[-1] == 3, \
        "Expected (T, H, W, C=3) array"
    assert video_array.dtype == np.uint8, "Expected uint8 array"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # temporal downsample
    frames = video_array[::downsample_temporal]
    
    # compute spatially downsampled size
    T, H, W, _ = frames.shape
    new_size = (W // downsample_spatial, H // downsample_spatial)
    
    # PNG compression param: 0 = no compression
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    
    for idx, frame in enumerate(frames):
        # frame is RGB; convert to BGR for OpenCV
        bgr = frame[..., ::-1]
        if downsample_spatial > 1:
            bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
        
        filename = os.path.join(output_dir, "frame_{:05d}.png".format(idx))
        success = cv2.imwrite(filename, bgr, png_params)
        if not success:
            raise RuntimeError("Failed to write frame ")
