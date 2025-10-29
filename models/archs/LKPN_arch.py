import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from xformers.ops import memory_efficient_attention
from transformers import PreTrainedModel, PretrainedConfig

# A helper for time embedding, typically used in diffusion models
# This encodes the timestep 't' into a high-dimensional vector
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        # Create sinusoidal positional embeddings for the timestep
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Pass through a linear layer as specified in the description
        embeddings = self.linear1(embeddings)
        embeddings = F.silu(embeddings)
        embeddings = self.linear2(embeddings)
        return embeddings

# A ResNet-style block with a residual connection
# It also incorporates the time embedding as a conditioning signal
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        # Shortcut for the residual connection
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        # Project and add time embedding
        # The description states it is added "before the out layer" of the ResBlock
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h + self.shortcut(x)

class SelfAttention(nn.Module):
    """
    An optimized self-attention layer for 2D feature maps that uses xformers
    for a highly memory-efficient and fast implementation.

    This version leverages a pre-compiled kernel from the xformers library,
    which is significantly more efficient than a manual PyTorch implementation
    for high-resolution inputs.
    """
    def __init__(self, channels):
        """
        Initializes the self-attention module.

        Args:
            channels (int): The number of input channels.
        """
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        
        # A single 1x1 convolution to generate Query, Key, and Value tensors.
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): The input feature map with shape (B, C, H, W).

        Returns:
            torch.Tensor: The output feature map with the same shape as x.
        """
        batch, channels, height, width = x.shape
        
        # 1. Normalize the input feature map.
        h = self.norm(x)
        
        # 2. Project input to Q, K, and V tensors.
        qkv = self.qkv(h)
        
        # Reshape to a sequence of tokens for the attention operation.
        qkv = qkv.view(batch, channels * 3, -1).permute(0, 2, 1).contiguous().half() # B, N, 3C

        # Split the combined QKV tensor into Q, K, and V.
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 3. Use xformers' memory-efficient attention.
        # This single function call handles scaling, softmax, and matrix multiplication
        # in a highly optimized way, avoiding the large intermediate attention matrix.
        h = memory_efficient_attention(q, k, v).float()
        
        # 4. Reshape the output back to the original 2D feature map dimensions.
        h = h.permute(0, 2, 1).view(batch, channels, height, width)
        
        # 5. Apply the output convolution and add the residual connection.
        h = self.out(h)
        return h + x
    

class SelfAttentionWindowed(nn.Module):
    """
    An optimized self-attention layer for 2D feature maps that uses a windowed
    approach combined with xformers' memory-efficient attention kernel.

    This hybrid approach provides the best of both worlds:
    1. It limits the attention to local windows, reducing complexity.
    2. It uses a highly optimized C++/CUDA kernel from xformers for the
       attention calculation itself, which is much faster than a manual
       PyTorch implementation.
    """
    def __init__(self, channels, window_size=16):
        """
        Initializes the self-attention module.

        Args:
            channels (int): The number of input channels.
            window_size (int): The size of the local window for attention.
                               The image will be divided into non-overlapping
                               windows of this size.
        """
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm = nn.GroupNorm(8, channels)
        
        # A single 1x1 convolution to generate Query, Key, and Value tensors.
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the self-attention layer.

        Args:
            x (torch.Tensor): The input feature map with shape (B, C, H, W).

        Returns:
            torch.Tensor: The output feature map with the same shape as x.
        """
        batch, channels, height, width = x.shape
        
        # 1. Normalize the input feature map.
        h = self.norm(x)

        # 2. Reshape the tensor into non-overlapping windows.
        # This creates a new dimension for the windows, allowing us to
        # compute attention within each window independently.
        h = einops.rearrange(
            h,
            'b c (h_w h) (w_w w) -> (b h_w w_w) c h w',
            h=self.window_size,
            w=self.window_size
        )
        
        # 3. Generate Q, K, and V tensors and apply attention using xformers.
        # The flattened window dimension acts as the batch dimension for xformers.
        qkv = self.qkv(h)
        
        # Reshape for memory_efficient_attention, which expects shape (B, N, C)
        qkv = qkv.view(qkv.shape[0], channels * 3, -1).permute(0, 2, 1).contiguous().half() # (num_windows, num_pixels_in_window, 3*channels)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Apply xformers' memory-efficient attention.
        h = memory_efficient_attention(q, k, v).float() # (num_windows, num_pixels_in_window, channels)

        # 4. Reshape the output back to the original dimensions.
        h = h.permute(0, 2, 1).view(
            batch,
            height // self.window_size,
            width // self.window_size,
            channels,
            self.window_size,
            self.window_size
        )
        h = einops.rearrange(
            h,
            'b h_w w_w c h w -> b c (h_w h) (w_w w)',
            h=self.window_size,
            w=self.window_size
        )
        
        # 5. Apply the output convolution and add the residual connection.
        h = self.out(h)
        return h + x

# The core building block of the UNet, combining ResBlock and SelfAttention
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.resblock = ResBlock(in_channels, out_channels, time_emb_dim)
        self.attention = SelfAttention(out_channels)
        # self.attention = SelfAttentionWindowed(out_channels)

    def forward(self, x, time_emb):
        x = self.resblock(x, time_emb)
        x = self.attention(x)
        return x

class LKPN_config(PretrainedConfig):
    def __init__(self, input_channels=4, k=5, time_emb_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.k = k
        self.time_emb_dim = time_emb_dim

# The main LKPN (Spatially-varying Kernel Prediction Network)
class LKPN_arch(PreTrainedModel):
    config_class = LKPN_config
    def __init__(self, config):
        super().__init__(config)
        input_channels=config.input_channels
        k=config.k
        time_emb_dim=config.time_emb_dim
        self.k = k
        self.input_channels = input_channels
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # The input is a concatenation of zt and zlq, doubling the channels
        self.initial_conv = nn.Conv2d(input_channels * 2, 64, kernel_size=3, padding=1)

        # Encoder layers with downsampling
        self.encoder1 = UNetBlock(64, 128, time_emb_dim)
        self.down1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        
        self.encoder2 = UNetBlock(128, 256, time_emb_dim)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        
        self.encoder3 = UNetBlock(256, 512, time_emb_dim)
        self.down3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        self.encoder4 = UNetBlock(512, 512, time_emb_dim)
        self.down4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck layer
        self.mid = UNetBlock(512, 512, time_emb_dim)

        # Decoder layers with upsampling
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.decoder4 = UNetBlock(1024, 512, time_emb_dim) # 512 (mid) + 512 (skip)
        
        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.decoder3 = UNetBlock(1024, 256, time_emb_dim) # 512 + 512 (skip)
        
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.decoder2 = UNetBlock(512, 128, time_emb_dim) # 256 + 256 (skip)
        
        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.decoder1 = UNetBlock(256, 64, time_emb_dim) # 128 + 128 (skip)
        
        # Final layers
        self.final_conv = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        
        # Final linear layer to estimate the spatially variant kernels
        self.kernel_estimation_layer = nn.Conv2d(input_channels, input_channels * k * k, kernel_size=1)

    def forward(self, zt, zlq, t):
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Concatenate inputs
        x = torch.cat([zt, zlq], dim=1)
        x = self.initial_conv(x)
        
        # Encoder
        e1 = self.encoder1(x, time_emb)
        d1 = self.down1(e1)
        
        e2 = self.encoder2(d1, time_emb)
        d2 = self.down2(e2)
        
        e3 = self.encoder3(d2, time_emb)
        d3 = self.down3(e3)
        
        e4 = self.encoder4(d3, time_emb)
        d4 = self.down4(e4)
        
        # Bottleneck
        m = self.mid(d4, time_emb)
        
        # Decoder with skip connections
        u4 = self.up4(m)
        c4 = torch.cat([u4, e4], dim=1)
        d4 = self.decoder4(c4, time_emb)

        u3 = self.up3(d4)
        c3 = torch.cat([u3, e3], dim=1)
        d3 = self.decoder3(c3, time_emb)
        
        u2 = self.up2(d3)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.decoder2(c2, time_emb)
        
        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.decoder1(c1, time_emb)

        # Final convolution to match original channel dimension
        final_output = self.final_conv(d1)
        
        # Final layer to transform into the spatially variant kernels
        kernels = self.kernel_estimation_layer(final_output)

        return kernels

class EfficientAffineConvolution(nn.Module):
    """
    Implements the Efficient Affine Convolution (EAC) operation.

    This module applies a spatially-variant convolution, where a unique
    k x k kernel is used for each pixel and channel of the input latent.
    It leverages `nn.functional.unfold` for an efficient, vectorized
    implementation without explicit loops.
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.r = (k - 1) // 2

    def forward(self, kernels, latent_z):
        """
        Applies the EAC operation.

        Args:
            kernels (torch.Tensor): Spatially-variant kernels from LKPN.
                                    Shape: (B, C, H, W, K, K)
            latent_z (torch.Tensor): The input latent tensor to be restored.
                                     Shape: (B, C, H, W)

        Returns:
            torch.Tensor: The restored latent tensor.
                          Shape: (B, C, H, W)
        """
        B, C, H, W = latent_z.shape

        # Step 1: Unfold the input latent into a collection of local patches.
        # This operation efficiently extracts all k x k neighborhoods of the
        # input latent, creating a tensor of shape (B, C * K * K, H * W).
        patches = F.unfold(latent_z, kernel_size=self.k, padding=self.r)

        # Step 2: Reshape the patches and kernels for efficient element-wise multiplication.
        # Patches are reshaped to (B, C, H*W, K*K).
        patches_reshaped = patches.view(B, C, H * W, self.k * self.k)

        # Kernels are reshaped from (B, C, H, W, K, K) to (B, C, H*W, K*K).
        kernels_reshaped = kernels.view(B, C, H * W, self.k * self.k)

        # Step 3: Perform element-wise multiplication followed by summation.
        # This operation is the core of the spatially-variant convolution.
        # For each pixel, the corresponding kernel is multiplied with the
        # extracted patch, and the results are summed to produce the output pixel.
        # The raw output has shape (B, C, H*W, K*K).
        output_raw = patches_reshaped * kernels_reshaped
        
        # The sum over the last dimension (K*K) yields the final output.
        # The shape is now (B, C, H*W).
        output_sum = output_raw.sum(dim=-1)

        # Step 4: Reshape the result back to the original image dimensions.
        # The final output has the same shape as the input latent.
        restored_latent = output_sum.view(B, C, H, W)

        return restored_latent

# Example usage
if __name__ == '__main__':
    # Assume input shapes for demonstration
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    kernel_size = 5
    timesteps = 1000

    # 1. Create dummy input latent and kernels from LKPN.
    # The LKPN output is (B, C, H, W, K, K)
    lkpn_kernels = torch.randn(batch_size, channels, height, width, kernel_size, kernel_size)
    zlq = torch.randn(batch_size, channels, height, width)

    # 2. Instantiate the EAC module with the kernel size.
    eac_module = EfficientAffineConvolution(k=kernel_size)
    
    # 3. Perform the EAC operation.
    restored_zlq = eac_module(lkpn_kernels, zlq)

    # 4. Verify the output shape.
    print(f"Input latent shape: {zlq.shape}")
    print(f"LKPN kernels shape: {lkpn_kernels.shape}")
    print(f"Restored latent shape: {restored_zlq.shape}")
    
    expected_shape = (batch_size, channels, height, width)
    assert restored_zlq.shape == expected_shape
    print(f"Shape matches expected output: {expected_shape}")


    ################################################################################
    # Assume input shapes for demonstration
    batch_size = 1
    channels = 4  # Corresponds to 'c'
    height = 512//8
    width = 512//8
    timesteps = 1000
    
    # Create dummy data
    zt = torch.randn(batch_size, channels, height, width)
    zlq = torch.randn(batch_size, channels, height, width)
    t = torch.randint(0, timesteps, (batch_size,))
    
    # Instantiate the model with k=5
    model = LKPN_arch(LKPN_config(input_channels=channels, k=5))
    
    # Run the forward pass
    output_kernels = model(zt, zlq, t)
    
    # Print the output shape to verify
    print(f"Output shape: {output_kernels.shape}")
    expected_channels = channels * 5 * 5
    print(f"Expected shape: ({batch_size}, {expected_channels}, {height}, {width})")
    assert output_kernels.shape == (batch_size, expected_channels, height, width)
    print("Shape matches expected output.")