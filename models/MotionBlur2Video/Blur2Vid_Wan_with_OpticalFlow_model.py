import cv2
import numpy as np
import torch
import os
from typing import List, Optional, Union
from tqdm import tqdm
from PIL import Image
import peft
import ptlflow
from ptlflow.utils import flow_utils

from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from diffusers import AutoencoderKLWan, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import free_memory

from models.base_model import BaseModel
from models.MotionBlur2Video.helpers import apply_lora_to_model, warp_tensor
from utils.loss import Loss
from utils import log_image, log_metrics
from utils.misc import log_video


def _prompt_clean(text: str) -> str:
    """Wan-style prompt cleaning (simplified without ftfy)."""
    import re

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def encode_video_wan(video: torch.Tensor, accelerator, vae, latents_mean, latents_std):
    """Encode video to Wan latent space. video: [B, F, C, H, W] -> [B, C, T, H, W] for VAE."""
    video = video.to(accelerator.device, dtype=vae.dtype)
    # Wan VAE expects [B, C, T, H, W]
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    out = vae.encode(video)
    if hasattr(out, "latent_dist"):
        latent = out.latent_dist.sample()
    else:
        latent = out.latents
    # Normalize like Wan pipeline: (latent - mean) * std
    latent = (latent - latents_mean) * latents_std
    return latent.to(memory_format=torch.contiguous_format)


def decode_latent_wan(latent: torch.Tensor, vae, latents_mean, latents_std):
    """Decode Wan latent to video. latent: [B, C, T, H, W], inverse norm then decode."""
    latent = latent / latents_std + latents_mean
    return vae.decode(latent).sample


def compute_prompt_embeds_wan(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    max_sequence_length: int,
    device,
    dtype,
    requires_grad: bool = False,
):
    """Encode prompt with UMT5 (Wan uses attention mask and last_hidden_state)."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [_prompt_clean(p) for p in prompt]
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    if requires_grad:
        prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
    else:
        with torch.no_grad():
            prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    # Pad/trim to fixed length like Wan pipeline
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]
            )
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds


def encode_image_wan(image_processor, image_encoder, image, device, dtype):
    """
    Encode image to CLIP embedding (Wan I2V uses hidden_states[-2]),
    following the official WanImageToVideoPipeline.encode_image behavior.

    Args:
        image: batch of images. If a tensor, it is expected to be in [-1, 1]
               with shape [B, C, H, W]. Otherwise, it can be a PIL image or
               a list of PIL / numpy images.
    """
    if torch.is_tensor(image):
        # [B, C, H, W] in [-1, 1] -> uint8 PIL images in [0, 255]
        if image.dim() == 4 and image.shape[1] == 3:
            image = image.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]
            image = [
                Image.fromarray(((x + 1.0) * 127.5).clip(0, 255).astype("uint8"))
                for x in image
            ]

    with torch.no_grad():
        # Match pipeline_wan_i2v: image_processor -> CLIPVisionModel(output_hidden_states=True)
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = image_encoder(**inputs, output_hidden_states=True)
        image_embeds = image_embeds.hidden_states[-2]

    return image_embeds.to(dtype=dtype, device=device)

def load_model_hook(models, input_dir):
    saved = {}
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()
        class_name = model._get_name()
        saved[class_name] = 1 if class_name not in saved.keys() else saved[class_name] + 1
        print(f"Loading model {class_name}_{saved[class_name]} from {input_dir}")
        
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
       

class Blur2Vid_Wan_OF_model(BaseModel):
    """Generate video from motion blurred image using Wan2.1 I2V with optical-flow-based noise warping."""

    def __init__(self, opt, logger):
        super(Blur2Vid_Wan_OF_model, self).__init__(opt, logger)
        self.load_handle.remove()
        self.load_handle = self.accelerator.register_load_state_pre_hook(load_model_hook)
        args = self.opt["args"]
        self.args = args
        model_id = getattr(
            args,
            "pretrained_model_name_or_path",
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        )

        # Wan2.1 I2V components
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            revision=getattr(args, "revision", None) or "main",
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            revision=getattr(args, "revision", None) or "main",
        )
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_id,
            subfolder="image_processor",
            revision=getattr(args, "revision", None) or "main",
        )
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model_id,
            subfolder="image_encoder",
            revision=getattr(args, "revision", None) or "main",
        )
        load_dtype = torch.bfloat16
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=getattr(args, "revision", None) or "main",
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            revision=getattr(args, "revision", None) or "main",
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            revision=getattr(args, "revision", None) or "main",
        )

        # Optical flow model (same as Blur2Vid_OF_model)
        opt_flow_model = ptlflow.get_model("sea_raft_l", ckpt_path="things")
        opt_flow_model.eval()
        self.opt_flow_model = opt_flow_model

        if getattr(args, "enable_slicing", False):
            self.vae.enable_slicing()
        if getattr(args, "enable_tiling", False):
            self.vae.enable_tiling()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.opt_flow_model.requires_grad_(False)

        self.vae_scale_factor_spatial = getattr(
            self.vae.config, "scale_factor_spatial", 8
        )
        self.vae_scale_factor_temporal = getattr(
            self.vae.config, "scale_factor_temporal", 4
        )
        self.model_config = self.transformer.config

        # Wan latent normalization (from pipeline prepare_latents)
        z_dim = getattr(self.vae.config, "z_dim", 16)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, z_dim, 1, 1, 1
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, z_dim, 1, 1, 1
        )

        if self.is_train:
            use_lora = getattr(args, "lora_finetune", True)
            if use_lora:
                # Freeze transformer and apply LoRA to reduce memory (fits ~80GB)
                self.transformer.requires_grad_(False)
                self.transformer.enable_gradient_checkpointing()
                lora_r = getattr(args, "lora_r", 8)
                lora_alpha = getattr(args, "lora_alpha", 16)
                lora_dropout = getattr(args, "lora_dropout", 0.0)
                lora_target_modules = getattr(
                    args,
                    "lora_target_modules",
                    ["to_q", "to_k", "to_v", "to_out", "add_k_proj", "add_v_proj"],
                )
                self.logger.info(
                    f"Applying LoRA to transformer (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})"
                )
                self.transformer = apply_lora_to_model(
                    self.transformer,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=lora_target_modules,
                    verbose=True,
                )
            else:
                self.transformer.requires_grad_(True)
                self.transformer.enable_gradient_checkpointing()
            self.init_training_settings()

        self.models.append(self.transformer)

        weight_dtype = torch.float32
        if self.accelerator.state.deepspeed_plugin:
            if "bf16" in getattr(
                self.accelerator.state.deepspeed_plugin.deepspeed_config or {},
                "bf16",
                {},
            ):
                weight_dtype = torch.bfloat16
            elif "fp16" in getattr(
                self.accelerator.state.deepspeed_plugin.deepspeed_config or {},
                "fp16",
                {},
            ):
                weight_dtype = torch.float16
        else:
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.image_encoder.to(self.accelerator.device, dtype=torch.float32)
        self.transformer.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=torch.float32)
        self.opt_flow_model.to(self.accelerator.device, dtype=torch.float32)
        self.latents_mean = self.latents_mean.to(self.accelerator.device)
        self.latents_std = self.latents_std.to(self.accelerator.device)

    def init_training_settings(self):
        self.transformer.train()
        train_opt = self.opt["train"]
        self.criterion = Loss(train_opt.loss).to(self.accelerator.device)

    def feed_data(self, data, is_train=True):
        self.sample = data
        if self.opt.train.patched if is_train else self.opt.val.patched:
            raise NotImplementedError()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self.optimizer_g = torch.optim.Adam(
            [{"params": optim_params}],
            lr=train_opt.optim.learning_rate,
            betas=(train_opt.optim.adam_beta1, train_opt.optim.adam_beta2),
            weight_decay=train_opt.optim.adam_weight_decay,
            eps=train_opt.optim.adam_epsilon,
        )
        self.optimizers.append(self.optimizer_g)

    def calculate_optical_flow_from_video(self, video: torch.Tensor) -> torch.Tensor:
        """Compute dense optical flow between consecutive frames of a video tensor [B, F, C, H, W]."""
        flows = []
        for i in range(video.shape[1] - 1):
            gt_flows = self.opt_flow_model(
                {"images": video[:, i : i + 2].to(self.opt_flow_model.dtype)}
            )
            flows.append(gt_flows["flows"][:, 0])
        flows = torch.stack(
            [torch.zeros_like(flows[0])] + flows, dim=1
        )  # [B, F, 2, H_f, W_f]
        return flows

    def convert_flow_to_video(self, flows: torch.Tensor) -> torch.Tensor:
        """Convert flow fields [B, F, 2, H, W] to RGB visualization [B, F, 3, H, W]."""
        gt_flow_video = []
        for i in range(flows.shape[1]):
            gt_flow_video.append(
                flow_utils.flow_to_rgb(flows[:, i].to(torch.float32)).to(
                    self.opt_flow_model.dtype
                )
            )
        gt_flow_video = torch.stack(gt_flow_video, dim=1)
        return gt_flow_video

    def optimize_parameters(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        with torch.no_grad():
            gt_video = self.sample["video"].to(
                memory_format=torch.contiguous_format
            ).float()
            blur_img = self.sample["blur_img"].to(
                memory_format=torch.contiguous_format
            ).float()
            prompts = self.sample["caption"]
            batch_size = len(prompts)
            guidance_mask = (
                torch.rand(batch_size, device=self.accelerator.device) >= 0.2
            )
            per_sample_prompts = [
                prompts[i] if guidance_mask[i] else "" for i in range(batch_size)
            ]

            # Encode GT video to normalized latent [B, C, T, H, W] (e.g. 16 channels)
            model_input_16 = encode_video_wan(
                gt_video, self.accelerator, self.vae, self.latents_mean, self.latents_std
            )
            model_input_16 = model_input_16.to(dtype=self.weight_dtype)

            # Build Wan-style latent condition from the blur image encoded as a video.
            # Use first frame as blur, remaining frames zeroed, then VAE-encode.
            B, F, C, H, W = gt_video.shape
            video_condition = torch.zeros_like(gt_video)
            video_condition[:, 0] = blur_img[:, 0]
            latent_condition = encode_video_wan(
                video_condition,
                self.accelerator,
                self.vae,
                self.latents_mean,
                self.latents_std,
            )
            latent_condition = latent_condition.to(dtype=self.weight_dtype)
            _, z_dim, T, H_l, W_l = latent_condition.shape

            # Construct a simple first-frame mask at latent resolution, with the
            # number of mask channels inferred from the transformer config:
            # in_channels = z_dim (noisy latents) + (mask_channels + z_dim) (condition).
            mask_channels = max(self.model_config.in_channels - 2 * z_dim, 0)
            if mask_channels > 0:
                mask_lat_size = torch.ones(
                    (B, 1, T, H_l, W_l),
                    device=latent_condition.device,
                    dtype=self.weight_dtype,
                )
                mask_lat_size[:, :, 1:] = 0
                mask_lat_size = mask_lat_size.repeat(
                    1, mask_channels, 1, 1, 1
                )  # mask_channels = 4
                condition = torch.cat([mask_lat_size, latent_condition], dim=1)
            else:
                condition = latent_condition

            # Blur -> CLIP image embedding (expect blur_img in [-1, 1])
            blur_for_clip = blur_img[:, 0]  # [B, C, H, W]
            image_embeds = encode_image_wan(
                self.image_processor,
                self.image_encoder,
                blur_for_clip,
                self.accelerator.device,
                self.weight_dtype,
            )

            prompt_embeds = compute_prompt_embeds_wan(
                self.tokenizer,
                self.text_encoder,
                per_sample_prompts,
                getattr(self.model_config, "max_text_seq_length", 512),
                self.accelerator.device,
                self.weight_dtype,
                requires_grad=False,
            )

            # Optical flow on GT video (for Go-with-the-Flow style noise warping)
            flows = self.calculate_optical_flow_from_video(gt_video)

        # Flow matching in VAE latent space (z_dim channels for latents only):
        # x_t = (1-t)*x_0 + t*noise, velocity = noise - x_0
        noise_16 = torch.randn_like(
            model_input_16, device=model_input_16.device, dtype=model_input_16.dtype
        )

        # Warp noise temporally using optical flow (operate on [B, T, C, H, W])
        noise_time_major = noise_16.permute(0, 2, 1, 3, 4)
        noise_time_major = warp_tensor(noise_time_major, flows)
        noise_16 = noise_time_major.permute(0, 2, 1, 3, 4)

        batch_size, num_channels, num_frames, height, width = model_input_16.shape

        timesteps = torch.rand(
            batch_size, device=model_input_16.device, dtype=model_input_16.dtype
        )  # [B]
        timesteps_reshaped = timesteps.view(batch_size, 1, 1, 1, 1)  # [B,1,1,1,1]
        x_t_latents = (1 - timesteps_reshaped) * model_input_16 + timesteps_reshaped * noise_16
        velocity_target = (
            noise_16 - model_input_16
        )  # [B, z_dim, T, H, W]; transformer predicts this

        # Concatenate noisy latents with condition channels to match Wan I2V pipeline.
        x_t = torch.cat([x_t_latents, condition], dim=1)  # [B, in_channels, T, H, W]

        # Scale timestep to scheduler range if needed (e.g. 0-999). Keep it 1D for the transformer.
        num_train_timesteps = getattr(self.scheduler.config, "num_train_timesteps", 1000)
        timestep_int = (timesteps * (num_train_timesteps - 1)).long().clamp(
            0, num_train_timesteps - 1
        )

        # Wan transformer: hidden_states [B, C, T, H, W], outputs [B, C, T, H, W] (velocity in VAE space)
        model_output = self.transformer(
            hidden_states=x_t,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_int,
            encoder_hidden_states_image=image_embeds,
            return_dict=False,
        )[0]

        # Flow matching loss: predict velocity
        loss = torch.nn.functional.mse_loss(model_output.float(), velocity_target.float())
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), 1.0)

        for opt in self.optimizers:
            opt.step()

        return {"all": loss.detach(), "diff": loss.detach()}

    @torch.no_grad()
    def validation(self):
        
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()

        from diffusers import WanImageToVideoPipeline

        self.pipeline = WanImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            image_processor=self.image_processor,
            image_encoder=self.image_encoder,
            transformer=self.accelerator.unwrap_model(self.transformer),
        )
        self.pipeline.to(self.accelerator.device, dtype=self.weight_dtype)

        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx)
            if idx >= self.max_val_steps:
                break
        for model in self.models:
            model.train()
        free_memory()

    def validate_step(self, _batch, idx):
        self.feed_data(_batch, is_train=False)
        output_dir = os.path.join(self.opt.path.experiments_root, "outputs")
        # Blur image for pipeline: [1, 1, C, H, W] -> PIL or numpy for first sample
        blur_np = (self.sample["blur_img"][0, 0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        frame = Image.fromarray(blur_np)

        assert len(self.sample["file_name"]) == 1
        filename = self.sample["file_name"][0]
        name = os.path.basename(os.path.dirname(os.path.splitext(filename)[0]))
        

        # Log GT and blur
        gt_video = self.sample["video"][0].permute(0, 2, 3, 1).cpu().numpy()
        gt_video = ((gt_video + 1) * 127.5) / 255
        num_frames = gt_video.shape[0] #self.sample["num_frames"][0]
        # gt_video = gt_video[0:num_frames]
        log_video(self.opt, self.accelerator, gt_video, f"gt_{name}", self.global_step)
        log_image(self.opt, self.accelerator, np.clip(blur_np.astype(np.float32) / 255, 0, 1).transpose(2, 0, 1)[np.newaxis], f"blur_{name}", self.global_step)

        video_latent = encode_video_wan(self.sample["video"], self.accelerator, self.vae, self.latents_mean, self.latents_std).float()#.to(dtype=self.weight_dtype)
        flow = self.calculate_optical_flow_from_video(self.sample["video"])
        warped = warp_tensor(torch.randn_like(video_latent), flow)
        gt_flow_video = self.convert_flow_to_video(flow)
        for i in range(flow.shape[0]):
            log_video(self.opt, self.accelerator, gt_flow_video[i].detach().cpu().float().numpy().transpose((0,2,3,1)), f'val_gtflow_{name}', self.global_step)
            log_video(self.opt, self.accelerator, warped[i].detach().cpu().float().numpy()[:,:3].transpose((0,2,3,1)), f'val_warped_{name}', self.global_step)
        
        del video_latent
        
        # Wan I2V: num_frames must follow 4*k+1 for temporal compression
        num_frames_wan = (
            (num_frames - 1) // self.vae_scale_factor_temporal
        ) * self.vae_scale_factor_temporal + 1
        num_frames_wan = max(num_frames_wan, 1)

        generator = (
            torch.Generator(device=self.accelerator.device).manual_seed(self.args.seed)
            if getattr(self.args, "seed", None)
            else None
        )
        out = self.pipeline(
            image=frame,
            prompt=getattr(self.args, "validation_prompt", "") or "",
            negative_prompt=getattr(self.args, "negative_prompt", "blur, low quality"),
            latents=warped,
            height=self.args.height,
            width=self.args.width,
            num_frames=num_frames_wan,
            num_inference_steps=getattr(self.args, "num_inference_steps", 50),
            guidance_scale=getattr(self.args, "guidance_scale", 5.0),
            generator=generator,
            output_type="np",
        )
        video = out.frames[0] if hasattr(out, "frames") else out[0]
        video = video[0:num_frames]
        os.makedirs(os.path.join(output_dir, "deblur_frames"), exist_ok=True)
        save_frames_as_pngs((video * 255).astype(np.uint8), os.path.join(output_dir, "deblur_frames", name))
        log_video(self.opt, self.accelerator, video, f"deblur_{name}", self.global_step)

        self.accelerator.wait_for_everyone()
        return idx + 1


def save_frames_as_pngs(video_array, output_dir, downsample_spatial=1, downsample_temporal=1):
    """Save (T, H, W, C) uint8 array as PNGs."""
    assert video_array.ndim == 4 and video_array.shape[-1] == 3
    assert video_array.dtype == np.uint8
    os.makedirs(output_dir, exist_ok=True)
    frames = video_array[::downsample_temporal]
    T, H, W, _ = frames.shape
    new_size = (W // downsample_spatial, H // downsample_spatial)
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    for idx, frame in enumerate(frames):
        bgr = frame[..., ::-1]
        if downsample_spatial > 1:
            bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
        path = os.path.join(output_dir, "frame_{:05d}.png".format(idx))
        if not cv2.imwrite(path, bgr, png_params):
            raise RuntimeError("Failed to write frame " + path)

