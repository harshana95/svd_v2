import glob
import cv2
import einops
from huggingface_hub import hf_hub_download
import numpy as np
import peft
import torch
import os
from typing import List, Optional, Union, Tuple
import torchvision
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file
import ptlflow
from ptlflow.utils import flow_utils

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

from models.MotionBlur2Video.helpers import random_insert_latent_frame, transform_intervals, apply_lora_to_model, warp_tensor
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


def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        # try:
        #     c = find_attr(_arch_modules, class_name)
        #     assert c is not None
        # except ValueError as e:  # class is not written by us. Try to load from diffusers
        #     print(f"Class {class_name} not found in archs. Trying to load from diffusers...")
        #     m = importlib.import_module('diffusers') # load the module, will raise ImportError if module cannot be loaded
        #     c = getattr(m, class_name)  # get the class, will raise AttributeError if class cannot be found    
        
        # load diffusers style into model
        folder = os.path.join(input_dir, f"{class_name}_{saved[class_name]}")
        combined_state_dict = {}
        # breakpoint()
        if "PeftModel" in class_name:
            model = peft.PeftModel.from_pretrained(model.base_model.model, folder, is_trainable=True)
            
        else:
            for file_path in glob.glob(os.path.join(folder, "*.safetensors")):
                state_dict_part = load_file(file_path)
                combined_state_dict.update(state_dict_part)
            try:
                model.load_state_dict(combined_state_dict)
            except Exception as e:
                print(f"{'='*50} Failed to load {class_name} {'='*50} {model} {e}")
        
        # load_model = c.from_pretrained(os.path.join(input_dir, f"{class_name}_{saved[class_name]}"))
        # model.load_state_dict(load_model.state_dict())
        # del load_model
        

class Blur2Deblur_model(BaseModel):
    """Generate video from blurred image. Video represents progressive deblurring"""

    def __init__(self, opt, logger):
        super(Blur2Deblur_model, self).__init__(opt, logger)
        self.load_handle.remove()
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        args = self.opt['args']
        self.args = args

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
        weight_path = hf_hub_download(
            repo_id='tedlasai/blur2vid', 
            filename="cogvideox-outsidephotos/checkpoint/model.safetensors"
        )
        transformer.load_state_dict(load_file(weight_path))
                
        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

        scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

        opt_flow_model = ptlflow.get_model('sea_raft_l', ckpt_path='things')
        opt_flow_model.eval()

        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

        # We only train the additional adapter controlnet layers
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        opt_flow_model.requires_grad_(False)
        
        self.vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
        self.model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.opt_flow_model = opt_flow_model
	
        # This drastically reduces VRAM usage by trading a small amount of compute.
        # It re-calculates activations during the backward pass instead of storing them.
        self.transformer.enable_gradient_checkpointing()
        
        # Enable gradient checkpointing for VAE to reduce memory during decode
        # This is especially useful when backpropagating through VAE decode for cycle loss
        self.vae.enable_gradient_checkpointing()
        if self.is_train:
            self.transformer.requires_grad_(False)
            lora_r = args.get("lora_r", 16)
            lora_alpha = args.get("lora_alpha", 32)
            lora_dropout = args.get("lora_dropout", 0.1)
            lora_target_modules = args.get("lora_target_modules", ["to_q", "to_k", "to_v"])
            
            print(f"Applying LoRA to generator model (r={lora_r}, alpha={lora_alpha})")
            self.transformer = apply_lora_to_model(
                self.transformer,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            
            self.init_training_settings()

        self.models.append(self.transformer)

        # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.state.deepspeed_plugin:
            # DeepSpeed is handling precision, use what's in the DeepSpeed config
            if ("fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
            ):
                weight_dtype = torch.float16
            if ("bf16" in self.accelerator.state.deepspeed_plugin.deepspeed_config
                and self.accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
            ):
                weight_dtype = torch.float16
        else:
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        # move frozen module to save VRAM
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.opt_flow_model.to(self.accelerator.device, dtype=torch.float32)
        # self.transformer.to(self.accelerator.device, dtype=weight_dtype)

    def init_training_settings(self):
        self.transformer.train()
        train_opt = self.opt['train']
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        if self.opt.train.patched:
            raise NotImplementedError()
            # self.grids(keys=[lq_key, gt_key], opt=self.opt.train if is_train else self.opt.val)

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
    def calculate_optical_flow_from_video(self, video):
        flows = []
        for i in range(video.shape[1]-1):
            gt_flows = self.opt_flow_model({'images': video[:, i:i+2].to(self.opt_flow_model.dtype)})
            flows.append(gt_flows['flows'][:, 0])
        flows =  torch.stack([torch.zeros_like(flows[0])] + flows, dim=1)#.to(dtype=self.weight_dtype)
        return flows

    def convert_flow_to_video(self, flows):
        gt_flow_video = []
        for i in range(flows.shape[1]):
            gt_flow_video.append(flow_utils.flow_to_rgb(flows[:, i].to(torch.float32)).to(self.opt_flow_model.dtype))
        gt_flow_video =  torch.stack(gt_flow_video, dim=1)#.to(dtype=self.weight_dtype)
        return gt_flow_video

    def optimize_parameters(self):
        DEBUG = True
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        with torch.no_grad():
            # breakpoint()
            gt_key = self.dataloader.dataset.gt_key
            lq_key = self.dataloader.dataset.lq_key
            
            gt_video = self.sample[lq_key].to(memory_format=torch.contiguous_format).float()
            blurred = gt_video[:, -2:-1]
            
            model_input = encode_video(gt_video, self.accelerator, self.vae).float()#.to(dtype=self.weight_dtype)  # [B, F, C, H, W]
            image_latent = encode_video(blurred, self.accelerator, self.vae).float()#.to(dtype=self.weight_dtype)  # [B, F, C, H, W]
            
            prompts = ["" for _ in range(batch_size)]
            input_intervals = torch.tensor([[-0.5, 0.5]], dtype=torch.float).to(memory_format=torch.contiguous_format).float()
            input_intervals = einops.repeat(input_intervals, 'N L -> B N L', B=batch_size)
            W = gt_video.shape[1]
            output_intervals = torch.tensor([[-0.5 + i * (1/W), -0.5 + (i + 1) * (1/W)] for i in range(W)]).to(memory_format=torch.contiguous_format).float()
            output_intervals = einops.repeat(output_intervals, 'N L -> B N L', B=batch_size)

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
        
        # # calculate oracle optical flow
        # flows = self.calculate_optical_flow_from_video(gt_video)
        # # warp noise
        # noise = warp_tensor(noise, flows)
                

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
        velocity = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)  # scheduler prediction_type is v_prediction
        
        # NOTE: Numerically stabilize the weighting. When alpha_cumprod ~ 1 (early timesteps),
        # (1 - alpha_cumprod) can be ~0 which explodes weights and can lead to inf/NaN gradients.
        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps].to(dtype=torch.float32)
        denom = (1.0 - alphas_cumprod).clamp(min=1e-4)
        weights = (1.0 / denom).to(dtype=velocity.dtype)
        while len(weights.shape) < len(velocity.shape):
            weights = weights.unsqueeze(-1)

        # is velocity == target? target is gt_video in latent space (first frame is the condition and it is skipped in the loss)
        # Compute diffusion loss in fp32 to avoid overflow in mixed precision
        vel_err = (velocity[~condition_mask].float() - target[~condition_mask].float())
        loss = torch.mean((weights.float() * (vel_err**2)).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
 
        # # regularization
        # # calculate cycle concistency loss in the latent space?
        # # VAE compression = 4x 8x 8x for frame, height, width channels increase 3->16. Overall ~48x compression
        # # thus, we need to upsample the velocity to the original resolution
        # #                          B   F  C   H    W    ->               B  F  C   H    W
        # # self.sample["blur_img"] [4,  1, 3, 720, 1280] -> image_latent [4, 1, 16, 90, 160]
        # # self.sample["video"]    [4, 17, 3, 720, 1280] -> model_input  [4, 5, 16, 90, 160]
        # # noisy_model_input, target, model_output, velocity             [4, 6, 16, 90, 160]
        # # condition_mask [4, 6]
        
        # dec = self.vae.decode(velocity[~condition_mask].view(batch_size, -1, num_channels, height, width).permute(0,2,1,3,4).to(self.weight_dtype) / self.vae.config.scaling_factor).sample
        
        # # # Decode in chunks to reduce memory usage during backprop
        # # velocity_latents = velocity[~condition_mask].view(batch_size, -1, num_channels, height, width).permute(0,2,1,3,4).to(torch.bfloat16) / self.vae.config.scaling_factor
        # # num_frames_to_decode = velocity_latents.shape[2]
        # # chunk_size = 2  # Decode 2 frames at a time to reduce memory
        
        # dec = dec.permute(0,2,1,3,4)
        # # [4, 6, 16, 90, 160] -> [4, 17, 3, 720, 1280]  range[-1,1]
        # dec = dec*0.5+0.5 # [-1,-1] -> [0,1]
        # dec = torch.clamp(dec, min=0.0, max=1.0)
        
        # # averaging the frames in linear space
        # gamma = 2.2
        # dec_lin = torch.pow(dec, gamma)
        # dec_avg_lin = torch.mean(dec_lin, dim=1, keepdim=True)
        # # gamma-encode back to sRGB domain
        # reblurred = torch.pow(dec_avg_lin, 1.0 / gamma)*2-1

        if DEBUG:
            # assert torch.isfinite(dec).all()
            # assert torch.isfinite(reblurred).all()
            # gt_flow_video = self.convert_flow_to_video(flows)
            if self.global_step % 100 == 0 or self.global_step <= 1:
                for i in range(batch_size):
                    # breakpoint()
                    log_video(self.opt, self.accelerator, noise[i,:,:3].detach().cpu().float().numpy().transpose((0,2,3,1)), f'train_warped_{i}', self.global_step)
                    
                    # log_video(self.opt, self.accelerator, gt_flow_video[i].detach().cpu().float().numpy().transpose((0,2,3,1)), f'train_gtflow_{i}', self.global_step)
                    log_video(self.opt, self.accelerator, gt_video[i].detach().cpu().float().numpy().transpose((0,2,3,1))*0.5+0.5, f'train_gt_{i}', self.global_step)
                    # log_video(self.opt, self.accelerator, dec[i].detach().cpu().float().numpy().transpose((0,2,3,1)), f'train_dec_{i}', self.global_step)
                    # log_image(self.opt, self.accelerator, np.clip(reblurred.detach().cpu().to(torch.float32).numpy()*0.5+0.5, 0,1)[i:i+1,0], f'reblur_{i}', self.global_step)
                    log_image(self.opt, self.accelerator, np.clip(blurred.detach().cpu().to(torch.float32).numpy()*0.5+0.5, 0,1)[i:i+1,0], f'train_blur_{i}', self.global_step)
            
        
        # # breakpoint()
        # # calculate cycle loss
        # loss_cycle = torch.mean((reblurred - self.sample["blur_img"])**2)
        # assert torch.isfinite(loss_cycle).all()

        # # loss_cycle = loss
        # loss_all = 0.1*loss_cycle + loss
        
        loss_all = loss
        self.accelerator.backward(loss_all)

        # Prevent parameter blow-ups (NaNs in model_output on next step usually come from inf/NaN grads here)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), 0.1)

        # # Hard stop if anything went non-finite during backward (prevents corrupting optimizer state/weights).
        # # This also helps pinpoint whether NaNs originate from forward vs backward.
        # if DEBUG:
            
        #     bad_grad = None
        #     for n, p in self.transformer.named_parameters():
        #         if p.grad is not None and not torch.isfinite(p.grad).all():
        #             bad_grad = n
        #             # self.logger.error(f"Non-finite grad detected in transformer param: {bad_grad}") 
        #             break # all NaN
        #     if bad_grad is not None:
        # # if not torch.isfinite(self.transformer.patch_embed.blur_condition.grad).all():
        #         self.logger.error(f"Non-finite grad detected at step {self.global_step}. Skipping optimizer step.")
        #         for opt in self.optimizers:
        #             opt.zero_grad(set_to_none=True)
        #         return {"all": loss_all.detach(), "diff": loss.detach(), 
        #         # "cycle": loss_cycle.detach()
        #         }

        
        # update weights
        for i in range(len(self.optimizers)):
            self.optimizers[i].step()
        return {
            'all': loss_all,
            'diff': loss,
            # 'cycle': loss_cycle
        }

    @torch.no_grad()
    def validation(self):
        torch.cuda.empty_cache()
        # return
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
        lq_key = self.dataloader.dataset.lq_key
        output_dir = os.path.join(self.opt.path.experiments_root, 'outputs')
        frame = ((self.sample[lq_key][:, -2:-1][:, 0].permute(0,2,3,1).cpu().numpy() + 1)*127.5).astype(np.uint8)
        name = f"{idx:05d}"

        #save the gt_video output
        gt_video = self.sample[lq_key][0].permute(0,2,3,1).cpu().numpy()
        gt_video = ((gt_video + 1) * 127.5)/255
        # save_frames_as_pngs((gt_video*255).astype(np.uint8), os.path.join(output_dir, "gt_frames", name))
        log_video(self.opt, self.accelerator, gt_video, f'gt_{name}', self.global_step)

        #save the blurred image
        log_image(self.opt, self.accelerator, np.clip(frame.astype(np.float32)/255, 0,1).transpose(0,3,1,2), f'blur_{name}', self.global_step)

        # calculate flow
        # video_latent = encode_video(self.sample["video"], self.accelerator, self.vae).float()#.to(dtype=self.weight_dtype)
        # shape = (
        #     frame.shape[0],
        #     (num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1,
        #     16,
        #     self.args.height // self.pipeline.vae_scale_factor_spatial,
        #     self.args.width // self.pipeline.vae_scale_factor_spatial,
        # )

        # flow = self.calculate_optical_flow_from_video(self.sample["video"])
        # warped = warp_tensor(torch.randn_like(video_latent), flow)
        # gt_flow_video = self.convert_flow_to_video(flow)
        # for i in range(flow.shape[0]):
        #     log_video(self.opt, self.accelerator, gt_flow_video[i].detach().cpu().float().numpy().transpose((0,2,3,1)), f'val_gtflow_{name}', self.global_step)
        #     log_video(self.opt, self.accelerator, warped[i].detach().cpu().float().numpy()[:,:3].transpose((0,2,3,1)), f'val_warped_{name}', self.global_step)
        # del video_latent
        warped = None
        
        input_intervals = torch.tensor([[-0.5, 0.5]], dtype=torch.float).to(memory_format=torch.contiguous_format).float()
        input_intervals = einops.repeat(input_intervals, 'N L -> B N L', B=1)
        W = gt_video.shape[1]
        output_intervals = torch.tensor([[-0.5 + i * (1/W), -0.5 + (i + 1) * (1/W)] for i in range(W)]).to(memory_format=torch.contiguous_format).float()
        output_intervals = einops.repeat(output_intervals, 'N L -> B N L', B=1)
        pipeline_args = {
                            "prompt": "",
                            "negative_prompt": "",
                            "image": frame,
                            "latents": warped,
                            "input_intervals": input_intervals,
                            "output_intervals": output_intervals,
                            "guidance_scale": self.args.guidance_scale,
                            "use_dynamic_cfg": self.args.use_dynamic_cfg,
                            "height": self.args.height,
                            "width": self.args.width,
                            "num_frames": self.args.max_num_frames,
                            "num_inference_steps": self.args.num_inference_steps,
                        }
        
        

        videos = log_validation(
            pipe=self.pipeline,
            args=self.args,
            accelerator=self.accelerator,
            pipeline_args=pipeline_args
        )

        #save the output video frames as pngs (uncompressed results) and mp4 (compressed results easily viewable)
        video = videos[0]
        save_frames_as_pngs((video*255).astype(np.uint8), os.path.join(output_dir, "deblur_frames", name))
        log_video(self.opt, self.accelerator, video, f'deblur_{name}', self.global_step)
        
        self.accelerator.wait_for_everyone()
        return idx+1
    
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
