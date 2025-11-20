import math
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 2-D Tensor of shape (batch_size, n) representing the input coordinates.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [batch_size x dim] Tensor of positional embeddings.
    """
    # Assuming timesteps is (batch_size, n)
    batch_size, input_coords_dim = timesteps.shape
    
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(device=timesteps.device)
    
    args = timesteps.float().unsqueeze(-1) * freqs # (batch_size, n, half_dim)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).view(batch_size, -1) # (batch_size, 4 * dim)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(TimestepBlock):
    def __init__(self, c, emb_channels, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.emb_channels = emb_channels
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, c),
        )

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.bb = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gg = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, emb):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.bb

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        
        # emb before residual connection
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x + emb_out

        return y + x * self.gg


class NAFNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], t_dim=4, **kwargs):
        super().__init__()
        self.model_channels = width # check if enough.
        time_embed_dim = self.model_channels * 4  # this will be reduced to # of block channel at each block
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels * t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                TimestepEmbedSequential(
                    *[NAFBlock(chan, time_embed_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # channel expander by 2
        self.latent_chan = chan
        self.channel_expand = nn.Conv2d(chan, chan*2, 1, bias=False)
        self.middle_blks = \
            TimestepEmbedSequential(
                *[NAFBlock(chan*2, time_embed_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                TimestepEmbedSequential(
                    *[NAFBlock(chan, time_embed_dim) for _ in range(num)]
                )
            )
        
        if out_channel == in_channel:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        
        self.padder_size = 2 ** len(self.encoders)

    def encode(self, x, emb):
        # encs = [x]
        x = self.intro(x)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, emb)
            # encs.append(x)
            x = down(x)

        x = self.channel_expand(x)
        x = self.middle_blks(x, emb)

        return x
    def sample(self, latents):
        # sample from mean and variance
        mean = latents[:, :self.latent_chan]
        logvar = latents[:, self.latent_chan:]
        latents = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return latents

    def decode(self, x, emb):
        # encs = encs[::-1]
        # inp = encs.pop()
        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            # x = x + enc_skip
            x = decoder(x, emb)

        x = self.ending(x)
        # x = x + self.skip_connection(inp)
        return x
    
    def get_embedding(self, t, dtype):
        emb = self.time_embed(timestep_embedding(t, self.model_channels)).type(dtype)
        return emb
    
    def forward(self, inp, timesteps=None):
        B, C, H, W = inp.shape
        if timesteps is None:
            timesteps = torch.tensor([0]*len(inp), dtype=torch.float32, device=inp.device)

        inp = self.check_image_size(inp)
        emb = self.get_embedding(timesteps, inp.dtype)

        z = self.encode(inp, emb)
        z_sample = self.sample(z)
        x = self.decode(z_sample, emb) 
        
        return x[:, :, :H, :W], z

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class PSF_Emb_config(PretrainedConfig):
    model_type = "PSF_Emb_arch"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PSF_Emb_arch(PreTrainedModel):
    config_class = PSF_Emb_config

    def __init__(self, config):
        super().__init__(config)
        self.model = NAFNet(**config.to_dict())
        self.init_weights()

    def forward(self, x, emb, *args):
        if len(args) > 0:
            x = torch.cat([x, *args], dim=1)
        return self.model(x, emb)

    def get_embedding(self, t, dtype):
        return self.model.get_embedding(t, dtype)

    def encode(self, x, emb):
        return self.model.encode(x, emb)
    
    def sample(self, latents):
        return self.model.sample(latents)

    def decode(self, latents, emb):
        return self.model.decode(latents, emb)