import torch 
import torch.nn.functional as F
import math
import random
import numpy as np
from PIL import Image

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.nn as nn
from typing import Optional

import torchvision


def apply_lora_to_model(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    bias: str = "none",
    verbose: bool = True,
) -> nn.Module:
    """
    Apply LoRA to a transformer model.
    
    Args:
        model: The transformer model (CausalWanModel or WanModel)
        r: LoRA rank (lower = less memory, less capacity). Default: 16
        lora_alpha: LoRA alpha scaling factor. Default: 32
        lora_dropout: LoRA dropout rate. Default: 0.1
        target_modules: List of module name patterns to apply LoRA to. 
                       PEFT uses pattern matching, so "q" will match "blocks.0.self_attn.q".
                       If None, defaults to attention layers: ["q", "k", "v", "o"]
        bias: Bias handling ("none", "all", "lora_only"). Default: "none"
        verbose: Print debug information. Default: True
    
    Returns:
        Model wrapped with LoRA adapters
    """
    if target_modules is None:
        # Default: target attention projections
        # PEFT will match these patterns in module names
        target_modules = ["q", "k", "v", "o"]
    
    if verbose:
        print(f"🔧 Applying LoRA with:")
        print(f"   Rank (r): {r}")
        print(f"   Alpha: {lora_alpha}")
        print(f"   Dropout: {lora_dropout}")
        print(f"   Target modules: {target_modules}")
        
        # Debug: show matching modules
        matching_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                for pattern in target_modules:
                    if pattern in name:
                        matching_modules.append(name)
                        break
        
        if matching_modules:
            print(f"   Found {len(matching_modules)} matching Linear layers for {target_modules}")
            if verbose and len(matching_modules) <= 20:
                for name in matching_modules[:10]:
                    print(f"     - {name}")
                if len(matching_modules) > 10:
                    print(f"     ... and {len(matching_modules) - 10} more")
        else:
            print("   ⚠️  Warning: No matching Linear layers found!")
            print("   Available Linear layers (first 10):")
            count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"     - {name}")
                    count += 1
                    if count >= 10:
                        break
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=matching_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=None,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    if verbose:
        print("\n📊 Trainable parameters:")
        model.print_trainable_parameters()
    
    return model

class ContinuousWarper:
    def __init__(self, height, width, device='cuda', dtype=torch.float32):
        self.height = height
        self.width = width
        self.device = device
        
        # 1. Initialize the Base Grid (Identity)
        # We keep this state to track the "total displacement" over time
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height, device=device),
                                        torch.arange(0, width, device=device),
                                        indexing='ij')
        
        # Stack to (H, W, 2)
        self.current_grid = torch.stack((grid_x, grid_y), dim=2).to(dtype)
        self.current_grid = self.current_grid.unsqueeze(0) # (1, H, W, 2)

    def step(self, original_image, flow):
        """
        Produce the next frame in the sequence.
        
        Args:
            original_image (Tensor): The purely original source image (N, C, H, W).
            flow (Tensor): The flow to apply for this step (N, 2, H, W).
                           (Can be static or changing per frame).
        """
        
        # Ensure flow is in (N, H, W, 2) format for grid math
        if flow.shape[1] == 2:
            flow = flow.permute(0, 2, 3, 1)

        # 2. Update the Grid (Lagrangian Integration)
        # Instead of moving pixels, we update the "lookup coordinates".
        # We subtract flow because we are looking *back* to where the pixel came from.
        # Note: If flow is defined in "screen space" (Eulerian), we sample the flow
        # at the *original* coordinates (simple subtraction).
        self.current_grid = self.current_grid - flow/2
        
        if original_image is None:
            return

        # 3. Normalize Grid to [-1, 1] for grid_sample
        # We create a temporary normalized grid for sampling
        norm_grid = self.current_grid.clone()
        norm_grid[..., 0] = 2.0 * norm_grid[..., 0] / max(self.width - 1, 1) - 1.0
        norm_grid[..., 1] = 2.0 * norm_grid[..., 1] / max(self.height - 1, 1) - 1.0
        
        # 4. Sample from the ORIGINAL image
        # Because we sample from original_image every time, we lose no quality.
        warped_frame = F.grid_sample(original_image, norm_grid, 
                                     mode='bilinear', padding_mode='reflection', align_corners=True)
        
        return warped_frame

def warp_tensor(noise, flows):
    batch_size, num_frames, num_channels, height, width = noise.shape
    # warp noise based on the optical flow
    warper = ContinuousWarper(height, width, noise.device, noise.dtype)
    with torch.no_grad():
        # resize optical flow to latent space 
        _b, _n, _c, _h, _w = flows.shape
        flows_resized = torchvision.transforms.functional.resize(flows.view(_b*_n, _c, _h, _w), (height, width)).view(_b, _n, _c, height, width)
        # subsample frames. we will do it using step
        # flows = flows[:, ::4]  # because vae compress frames by 4x
        init_noise = noise[:, 0]
        for i in range(0, _n):
            if i % (num_frames-1) == 0:
                warped_noise = warper.step(init_noise, flows_resized[:, i])
                noise[:, i//(num_frames-1)] = warped_noise
            else:
                warper.step(None, flows_resized[:, i])
    return noise

def random_insert_latent_frame(
    image_latent: torch.Tensor,
    noisy_model_input: torch.Tensor,
    target_latents: torch.Tensor,
    input_intervals: torch.Tensor,
    output_intervals: torch.Tensor,
    special_info
):
    """
    Inserts latent frames into noisy input, pads targets, and builds flattened intervals with flags.

    Args:
        image_latent:     [B, latent_count, C, H, W]
        noisy_model_input:[B, F, C, H, W]
        target_latents:   [B, F, C, H, W]
        input_intervals:  [B, N, frames_per_latent, L]
        output_intervals: [B, M, frames_per_latent, L]

    For each sample randomly choose:
    Mode A (50%):
        - Insert two image_latent frames at start of noisy input and targets.
        - Pad target_latents by prepending two zero-frames.
        - Pad input_intervals by repeating its last group once.
    Mode B (50%):
        - Insert one image_latent frame at start and repeat last noisy frame at end.
        - Pad target_latents by prepending one one-frame and appending last target frame.
        - Pad output_intervals by repeating its last group once.

    After padding intervals, flatten each group from [frames_per_latent, L] to [frames_per_latent * L],
    then append a 4-element flag (1 for input groups, 0 for output groups).

    Returns:
        outputs:     Tensor [B, F+2, C, H, W]
        new_targets: Tensor [B, F+2, C, H, W]
        masks:       Tensor [B, F+2] bool mask of latent inserts
        intervals:   Tensor [B, N+M+1, fpl * L + 4]
    """
    B, F, C, H, W = noisy_model_input.shape
    _, N, fpl, L = input_intervals.shape
    _, M, _, _ = output_intervals.shape
    device = noisy_model_input.device

    new_F = F + 1 if special_info == "just_one" else F + 2
    outputs = torch.empty((B, new_F, C, H, W), device=device)
    masks = torch.zeros((B, new_F), dtype=torch.bool, device=device)
    combined_groups = N + M #+ 1
    feature_len = fpl * L
    # intervals = torch.empty((B, combined_groups, feature_len + 4), device=device,
    #                         dtype=input_intervals.dtype)
    intervals = torch.empty((B, combined_groups, feature_len), device=device,
                            dtype=input_intervals.dtype)
    new_targets = torch.empty((B, new_F, C, H, W), device=device,
                            dtype=target_latents.dtype)

    for b in range(B):
        latent = image_latent[b, 0]
        frames = noisy_model_input[b]
        tgt = target_latents[b]

        limit = 10 if special_info == "use_a" else 0.5
        if special_info == "just_one": #ALWAYS_MODE_A
            # Mode A: two latent inserts, zero-prefixed targets
            outputs[b, 0] = latent
            masks[b, :1] = True
            outputs[b, 1:] = frames

            # pad targets: two large-numbers - these should be ignored
            large_number = torch.ones_like(tgt[0])*10000
            new_targets[b, 0] = large_number
            new_targets[b, 1:] = tgt

            # pad intervals: input + replicated last input group
            #pad_group = input_intervals[b, -1:].clone()
            in_groups = input_intervals[b] #torch.cat([input_intervals[b], pad_group], dim=0)
            out_groups = output_intervals[b]
        elif random.random() < limit: #ALWAYS_MODE_A
            # Mode A: two latent inserts, zero-prefixed targets
            outputs[b, 0] = latent
            outputs[b, 1] = latent
            masks[b, :2] = True
            outputs[b, 2:] = frames

            # pad targets: two large-numbers - these should be ignored
            large_number = torch.ones_like(tgt[0])*10000
            new_targets[b, 0] = large_number
            new_targets[b, 1] = large_number
            new_targets[b, 2:] = tgt

            # pad intervals: input + replicated last input group
            pad_group = input_intervals[b, -1:].clone()
            in_groups = torch.cat([input_intervals[b], pad_group], dim=0)
            out_groups = output_intervals[b]
        else:
            # Mode B: one latent insert & last-frame repeat, one-prefixed/appended targets
            outputs[b, 0] = latent
            masks[b, 0] = True
            outputs[b, 1:new_F-1] = frames
            outputs[b, new_F-1] = frames[-1]

            # pad targets: one one-frame then original then last frame
            zero = torch.zeros_like(tgt[0])
            new_targets[b, 0] = zero
            new_targets[b, 1:new_F-1] = tgt
            new_targets[b, new_F-1] = tgt[-1]

            # pad intervals: output + replicated last output group
            in_groups = input_intervals[b]
            pad_group = output_intervals[b, -1:].clone()
            out_groups = torch.cat([output_intervals[b], pad_group], dim=0)

        # flatten & flag groups
        flat_in = in_groups.reshape(-1, feature_len)
        proc_in = torch.cat([flat_in], dim=1)

        flat_out = out_groups.reshape(-1, feature_len)
        proc_out = torch.cat([flat_out], dim=1)

        intervals[b] = torch.cat([proc_in, proc_out], dim=0)

    return outputs, new_targets, masks, intervals




def transform_intervals(
    intervals: torch.Tensor,
    frames_per_latent: int = 4,
    repeat_first: bool = True
) -> torch.Tensor:
    """
    Pad and reshape intervals into [B, num_latent_frames, frames_per_latent, L].

    Args:
        intervals: Tensor of shape [B, N, L]
        frames_per_latent: number of frames per latent group (e.g., 4)
        repeat_first: if True, pad at the beginning by repeating the first row; otherwise pad at the end by repeating the last row.

    Returns:
        Tensor of shape [B, num_latent_frames, frames_per_latent, L]
    """
    B, N, L = intervals.shape
    num_latent = math.ceil(N / frames_per_latent)
    target_N = num_latent * frames_per_latent
    pad_count = target_N - N

    if pad_count > 0:
        # choose row to repeat
        pad_row = intervals[:, :1, :] if repeat_first else intervals[:, -1:, :]
        # replicate pad_row pad_count times
        pad = pad_row.repeat(1, pad_count, 1)
        # pad at beginning or end
        if repeat_first:
            expanded = torch.cat([pad, intervals], dim=1)
        else:
            expanded = torch.cat([intervals, pad], dim=1)
    else:
        expanded = intervals[:, :target_N, :]

    # reshape into latent-frame groups
    return expanded.view(B, num_latent, frames_per_latent, L)