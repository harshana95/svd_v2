import functools
import gc
import random
import einops
import numpy as np
import torch
from tqdm import tqdm

from transformers import PretrainedConfig, AutoTokenizer
from diffusers import get_scheduler, AutoencoderKL, \
    UNet2DConditionModel, EulerDiscreteScheduler, T2IAdapter

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from models.archs import define_network
from models.archs.LKPN_arch import EfficientAffineConvolution
from models.base_model import BaseModel
from pipelines.DiffusionWithLKPNPipeline import DiffusionSingleImageLKPNPipeline
from utils.dataset_utils import merge_patches
from utils.loss import Loss
from utils import log_image, log_metrics

#     ========================= TODO: NOT FINISHED

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    

class ResShift_model(BaseModel):

    def __init__(self, opt, logger):
        super(ResShift_model, self).__init__(opt, logger)
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        pretrained_model_name_or_path = self.opt.pretrained_model_name_or_path
        revision = self.opt.revision
        variant = self.opt.variant
        
        # Load scheduler and models
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
        )

        vae.requires_grad_(False)
        vae.to(self.accelerator.device)
        
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.models = [unet]

    def setup_optimizers(self):
        opt = self.opt.train.optim

        # Optimizer creation
        optimizer_class = torch.optim.AdamW
        params_to_optimize = list(self.t2iadapter_1.parameters())
        
        optimizer = optimizer_class(
            params_to_optimize,
            lr=opt.learning_rate,
            betas=(opt.adam_beta1, opt.adam_beta2),
            weight_decay=opt.adam_weight_decay,
            eps=opt.adam_epsilon,
            )
       
        self.optimizers.append(optimizer)
    
    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        if self.opt.train.patched:
            self.grids(keys=[lq_key, gt_key,], 
                       opt=self.opt.train if is_train else self.opt.val)
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def optimize_parameters(self):
        gt_key = self.dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key

        image_1 = self.sample[lq_key]
        gt_1 = self.sample[gt_key]

        # encode pixel values with batch size of at most 8 to avoid OOM
        gt_latents = []
        for i in range(0, gt_1.shape[0], 8):
            gt_latents.append(self.vae.encode(gt_1[i: i + 8]).latent_dist.sample())
        gt_latents = torch.cat(gt_latents, dim=0)
        gt_latents = gt_latents * self.vae.config.scaling_factor
        gt_latents = gt_latents.to(self.weight_dtype)

         # Sample noise that we'll add to the latents
        noise = torch.randn_like(gt_latents)
        bsz = gt_latents.shape[0]

        # Cubic sampling to sample a random timestep for each image.
        # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
        timesteps = torch.rand((bsz,), device=gt_latents.device)
        timesteps = (1 - timesteps ** 3) * self.noise_scheduler.config.num_train_timesteps
        timesteps = timesteps.long().to(self.noise_scheduler.timesteps.dtype)
        timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(gt_latents, noise, timesteps)

        # Scale the noisy latents for the UNet
        sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
        inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)  # these are around -4 to +4

        
        # get latents for img1 and img2
        z_1 = self.vae.encode(image_1).latent_dist.sample() * self.vae.config.scaling_factor
        
        # calculate latent kernels
        k_1 = self.lkpn(inp_noisy_latents, z_1, timesteps)

        # convolve latents with estimated kernels
        batch_size, channels, height, width = inp_noisy_latents.shape
        k_1 = k_1.view(batch_size, channels, self.lkpn.k, self.lkpn.k, height, width)
        k_1 = k_1.permute(0, 1, 4, 5, 2, 3)
        z_1_ref = self.eac(k_1, z_1)

        # LKPN loss
        loss_lkpn = torch.mean(torch.nn.functional.mse_loss(z_1_ref, gt_latents))  # latent space loss 
        # loss_lkpn += torch.mean(torch.nn.functional.mse_loss(self.vae.decode(z_1_ref / self.vae.config.scaling_factor).sample, gt_1))  # pixel space loss
        loss_lkpn *= 1
        self.accelerator.backward(loss_lkpn)
        if self.accelerator.sync_gradients:
            params_to_clip = list(self.lkpn.parameters())
            self.accelerator.clip_grad_norm_(params_to_clip, 1.0)
        self.optimizers[1].step()
        self.optimizers[1].zero_grad()
        z_1_ref = z_1_ref.detach()

        
        # concatenate latents and convolved latents and pass them throught T2I adapters
        down_block_additional_residuals_1x = self.t2iadapter_1(torch.cat([z_1, z_1_ref], dim=-3))

        # generate down block additional residuals by multiplying the outputs of two adapters
        down_block_additional_residuals = []
        for i in range(len(down_block_additional_residuals_1x)):
            down_block_additional_residuals.append(
                down_block_additional_residuals_1x[i].to(dtype=self.weight_dtype)
            )
        
        # Predict the noise residual
        model_pred = self.unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states=self.prompt_ids[:bsz],
            added_cond_kwargs={k: v[:bsz] for k, v in self.unet_added_conditions.items()},
            down_block_additional_residuals=down_block_additional_residuals,
            return_dict=False,
        )[0]

        # Denoise the latents
        denoised_latents = model_pred * (-sigmas) + noisy_latents
        weighing = sigmas ** -2.0

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            # target = latents  # we are computing loss against denoise latents
            target = gt_latents
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            # target = noise_scheduler.get_velocity(latents, noise, timesteps)
            target = self.noise_scheduler.get_velocity(gt_latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # diffusion MSE loss
        loss_diff = torch.mean(
            (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            dim=1,
        )
        loss_diff = loss_diff.mean()
        
        # total loss
        loss_all = loss_diff + loss_lkpn

        self.accelerator.backward(loss_diff)
        if self.accelerator.sync_gradients:
            params_to_clip = list(self.t2iadapter_1.parameters())
            self.accelerator.clip_grad_norm_(params_to_clip, 0.01)
        self.optimizers[0].step()
        self.optimizers[0].zero_grad()

        return {'all': loss_all, 'diff': loss_diff, 'lkpn': loss_lkpn}
    
    @torch.no_grad()
    def validation(self):
        gc.collect()
        torch.cuda.empty_cache()
        idx = 0
        for model in self.models:
            model.eval()
        noise_scheduler_tmp = EulerDiscreteScheduler.from_pretrained(self.opt.pretrained_model_name_or_path, subfolder="scheduler")
        self.pipeline = DiffusionSingleImageLKPNPipeline(
            self.vae,
            self.unet,
            self.lkpn,
            self.eac,
            self.t2iadapter_1,
            noise_scheduler_tmp,
            use_1_as_start=self.opt.val.use_1_as_start,
        )
        dataloader = DataLoader(Subset(self.dataloader.dataset, np.arange(5)), 
                                shuffle=False, 
                                batch_size=1)
        print(f"Tesing using {len(dataloader)} training data...")
        dataloader = self.accelerator.prepare(dataloader)
        for batch in dataloader:
            idx = self.validate_step(batch, idx, self.dataloader.dataset.lq_key, self.dataloader.dataset.gt_key)
        self.accelerator._dataloaders.remove(dataloader)
        for batch in tqdm(self.test_dataloader):
            idx = self.validate_step(batch, idx, self.test_dataloader.dataset.lq_key, self.test_dataloader.dataset.gt_key)
            

        for model in self.models:
            model.train()
            
    def validate_step(self, batch, idx,lq_key,gt_key):
        self.feed_data(batch, is_train=False)
        
        if self.opt.val.patched:
            b,c,h,w = self.original_size[lq_key]
            pred = []
            for _ in self.setup_patches():    
                image_1 = self.sample[lq_key]
                bsz = image_1.shape[0]  
                out = self.pipeline(
                    image_1,
                    prompt_embeds=self.prompt_ids[:bsz],
                    added_cond_kwargs={k: v[:bsz] for k, v in self.unet_added_conditions.items()},
                    rescale_image_a=1, 
                    rescale_image_b=1,
                    num_inference_steps=50,
                    output_intermediate_steps=True,
                    n_output_intermediate_steps=10,
                )
                pred.append(out.images)
            pred = torch.cat(pred, dim=0)
            pred = einops.rearrange(pred, '(b n) c h w -> b n c h w', b=b)
            out = []
            for i in range(len(pred)):
                merged = merge_patches(pred[i], self.sample[lq_key+'_patched_pos'])
                out.append(merged[..., :h, :w])
            lq1 = self.sample[lq_key+'_original']
            gt = self.sample[gt_key+'_original']
            out = torch.stack(out)
        else: 
            lq1 = self.sample[lq_key]
            gt = self.sample[gt_key]
            bsz = lq1.shape[0]  
            output = self.pipeline(
                lq1,
                prompt_embeds=self.prompt_ids[:bsz],
                added_cond_kwargs={k: v[:bsz] for k, v in self.unet_added_conditions.items()},
                rescale_image_a=1, 
                num_inference_steps=50,
                output_intermediate_steps=True,
                n_output_intermediate_steps=10,
            )
            out = output.images
            steps = output.step_outputs

        lq1 = lq1.cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.cpu().numpy()
        for i in range(len(gt)):
            idx += 1
            image1 = [lq1[i], gt[i], out[i]]
            for j in range(len(image1)):
                if image1[j].shape[0] == 1:
                    image1[j] = einops.repeat(image1[j], '1 h w -> 3 h w')
            image1 = np.stack(image1)
            image1 = np.clip(image1, 0, 1)
            log_image(self.opt, self.accelerator, image1, f'{idx:04d}', self.global_step)  # image format (N,C,H,W)
            log_image(self.opt, self.accelerator, np.clip(np.stack([out[i]]), 0,1), f'out_{idx:04d}', self.global_step)
            log_metrics(gt[i], out[i], self.opt.val.metrics, self.accelerator, self.global_step)

            # log intermediate steps
            image2 = []
            for step in steps:
                image2.append(step[i].cpu().numpy())
            image2 = np.stack(image2).clip(0, 1)
            log_image(self.opt, self.accelerator, image2, f'steps_{idx:04d}', self.global_step)  # image format (N,C,H,W)
            
        return idx
    