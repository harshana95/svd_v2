"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import torch
import torch.nn.functional as F

def apply_gains(bayer_images, red_gains, blue_gains):
  """Applies white balance gains to a batch of Bayer images."""
  # Assumes bayer_images shape is (N, H, W, 4)
  green_gains = torch.ones_like(red_gains)
  gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
  gains = gains[:, None, None, :]
  return bayer_images * gains


def demosaic(bayer_images):
  """Bilinearly demosaics a batch of RGGB Bayer images."""
  # Assumes bayer_images shape is (N, H, W, 4)
  # This implementation exploits how edges are aligned when upsampling.
  # This is a PyTorch translation of the original TensorFlow implementation.

  # All PyTorch image operations expect (N, C, H, W)
  # We permute to (N, 4, H, W)
  bayer_images = bayer_images.permute(0, 3, 1, 2)
  shape = bayer_images.shape
  H = shape[2]
  W = shape[3]

  # Red
  red = bayer_images[:, 0:1, :, :]
  red = F.interpolate(red, size=(H * 2, W * 2), mode='bilinear', align_corners=False)

  # Green (Red row)
  green_red = bayer_images[:, 1:2, :, :]
  green_red = torch.flip(green_red, dims=[3]) # Flip left-right
  green_red = F.interpolate(green_red, size=(H * 2, W * 2), mode='bilinear', align_corners=False)
  green_red = torch.flip(green_red, dims=[3]) # Flip left-right

  # Green (Blue row)
  green_blue = bayer_images[:, 2:3, :, :]
  green_blue = torch.flip(green_blue, dims=[2]) # Flip up-down
  green_blue = F.interpolate(green_blue, size=(H * 2, W * 2), mode='bilinear', align_corners=False)
  green_blue = torch.flip(green_blue, dims=[2]) # Flip up-down

  # Blue
  blue = bayer_images[:, 3:4, :, :]
  blue = torch.flip(blue, dims=[2, 3]) # Flip up-down and left-right
  blue = F.interpolate(blue, size=(H * 2, W * 2), mode='bilinear', align_corners=False)
  blue = torch.flip(blue, dims=[2, 3]) # Flip up-down and left-right

  # Reconstruct Green channel
  # We use PixelUnshuffle (space_to_depth) and PixelShuffle (depth_to_space)
  pixel_unshuffle = torch.nn.PixelUnshuffle(2)
  green_red_unshuffled = pixel_unshuffle(green_red)
  green_blue_unshuffled = pixel_unshuffle(green_blue)

  green_at_red = (green_red_unshuffled[:, 0] + green_blue_unshuffled[:, 0]) / 2
  green_at_green_red = green_red_unshuffled[:, 1]
  green_at_green_blue = green_blue_unshuffled[:, 2]
  green_at_blue = (green_red_unshuffled[:, 3] + green_blue_unshuffled[:, 3]) / 2

  green_planes = [
      green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
  ]
  green_stacked = torch.stack(green_planes, dim=1) # (N, 4, H, W)
  
  pixel_shuffle = torch.nn.PixelShuffle(2)
  green = pixel_shuffle(green_stacked) # (N, 1, H*2, W*2)

  # Concatenate channels
  rgb_images = torch.cat([red, green, blue], dim=1) # (N, 3, H*2, W*2)
  
  # Permute back to (N, H, W, C)
  rgb_images = rgb_images.permute(0, 2, 3, 1)
  return rgb_images


def apply_ccms(images, ccms):
  """Applies color correction matrices."""
  # images shape (N, H, W, 3)
  # ccms shape (N, 3, 3)
  
  # We want to perform a batched matrix-vector multiplication for each pixel.
  # We can use broadcasting and sum.
  # images[..., None, :] -> (N, H, W, 1, 3)
  # ccms[:, None, None, :, :] -> (N, 1, 1, 3, 3)
  # (N, H, W, 1, 3) * (N, 1, 1, 3, 3) -> (N, H, W, 3, 3)
  # sum(..., dim=-1) -> (N, H, W, 3)
  
  images_expanded = images[..., None, :]
  ccms_expanded = ccms[:, None, None, :, :]
  
  # This line is equivalent to tf.reduce_sum(images * ccms, axis=-1)
  # and performs the (image @ ccm.T) operation at each pixel.
  return torch.sum(images_expanded * ccms_expanded, dim=-1)


def gamma_compression(images, gamma=2.2):
  """Converts from linear to gamma space."""
  # Clamps to prevent numerical instability of gradients near zero.
  return torch.clamp(images, min=1e-8) ** (1.0 / gamma)


def process(bayer_images, red_gains, blue_gains, cam2rgbs):
  """Processes a batch of Bayer RGGB images into sRGB images."""
  # White balance.
  bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
  # Demosaic.
  bayer_images = torch.clamp(bayer_images, 0.0, 1.0)
  images = demosaic(bayer_images)
  # Color correction.
  images = apply_ccms(images, cam2rgbs)
  # Gamma compression.
  images = torch.clamp(images, 0.0, 1.0)
#   images = gamma_compression(images)
  return images


# ===================================================================== Unprocess functions


def random_ccm():
  """Generates random RGB -> Camera color correction matrices."""
  # Takes a random convex combination of XYZ -> Camera CCMs.
  xyz2cams_list = [[[1.0234, -0.2969, -0.2266],
                    [-0.5625, 1.6328, -0.0469],
                    [-0.0703, 0.2188, 0.6406]],
                   [[0.4913, -0.0541, -0.0202],
                    [-0.613, 1.3513, 0.2906],
                    [-0.1564, 0.2151, 0.7183]],
                   [[0.838, -0.263, -0.0639],
                    [-0.2887, 1.0725, 0.2496],
                    [-0.0627, 0.1427, 0.5438]],
                   [[0.6596, -0.2079, -0.0562],
                    [-0.4782, 1.3016, 0.1933],
                    [-0.097, 0.1581, 0.5181]]]
  num_ccms = len(xyz2cams_list)
  xyz2cams = torch.tensor(xyz2cams_list, dtype=torch.float32)
  
  # Generate random weights
  weights = (1e8 - 1e-8) * torch.rand(num_ccms, 1, 1) + 1e-8
  weights_sum = torch.sum(weights, dim=0)
  xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

  # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
  rgb2cam = torch.matmul(xyz2cam, rgb2xyz)

  # Normalizes each row.
  rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
  return rgb2cam


def random_gains():
  """Generates random gains for brightening and white balance."""
  # RGB gain represents brightening.
  rgb_gain = 1.0 / (torch.randn(()) * 0.1 + 0.8)

  # Red and blue gains represent white balance.
  red_gain = (2.4 - 1.9) * torch.rand(()) + 1.9
  blue_gain = (1.9 - 1.5) * torch.rand(()) + 1.5
  return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = torch.clamp(image, 0.0, 1.0)
  return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  # Clamps to prevent numerical instability of gradients near zero.
  return torch.clamp(image, min=1e-8) ** 2.2


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  shape = image.shape
  image = image.view(-1, 3)
  # The TF tensordot with axes=[[-1], [-1]] is equivalent to
  # matmul with the transpose of the second argument.
  image = torch.matmul(image, ccm.T)
  return image.view(shape)


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  green_gain = torch.tensor(1.0, dtype=red_gain.dtype, device=red_gain.device)
  
  gains = torch.stack([1.0 / red_gain, green_gain, 1.0 / blue_gain]) / rgb_gain
  gains = gains[None, None, :]

  # Prevents dimming of saturated pixels by smoothly masking gains near white.
  gray = torch.mean(image, dim=-1, keepdim=True)
  inflection = 0.9
  mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
  safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)
  return image * safe_gains


def mosaic(image):
  """Extracts RGGB Bayer planes from an RGB image."""
  shape = image.shape
  red = image[0::2, 0::2, 0]
  green_red = image[0::2, 1::2, 1]
  green_blue = image[1::2, 0::2, 1]
  blue = image[1::2, 1::2, 2]
  image = torch.stack((red, green_red, green_blue, blue), dim=-1)
  image = image.view(shape[0] // 2, shape[1] // 2, 4)
  return image


def unprocess(images):
  """Unprocesses an image from sRGB to realistic raw data."""
  # Randomly creates image metadata.
  rgb2cam = random_ccm()
  cam2rgb = torch.linalg.inv(rgb2cam)
  rgb_gain, red_gain, blue_gain = random_gains()
  if not isinstance(images, list):
    images = [images]
  
  for i in range(len(images)):
    image = images[i]
    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    image = torch.clamp(image, 0.0, 1.0)
    # Applies a Bayer mosaic.
    image = mosaic(image)
    images[i] = image

  metadata = {
      'cam2rgb': cam2rgb,
      'rgb_gain': rgb_gain,
      'red_gain': red_gain,
      'blue_gain': blue_gain,
  }
  return images, metadata


def random_noise_levels():
  """Generates random noise levels from a log-log linear distribution."""
  log_min_shot_noise = torch.log(torch.tensor(0.0001))
  log_max_shot_noise = torch.log(torch.tensor(0.012))
  log_shot_noise = (log_max_shot_noise - log_min_shot_noise) * torch.rand(()) + log_min_shot_noise
  shot_noise = torch.exp(log_shot_noise)

  line = lambda x: 2.18 * x + 1.20
  log_read_noise = line(log_shot_noise) + torch.randn(()) * 0.26
  read_noise = torch.exp(log_read_noise)
  return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  stddev = torch.sqrt(variance)
  noise = torch.randn_like(image) * stddev
  return image + noise