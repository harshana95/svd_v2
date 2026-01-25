import torch
import torch.nn as nn
import torchvision.models as models
from transformers import PreTrainedModel, PretrainedConfig

class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and a skip connection.
    As described in the paper, it contains only one ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False) # bias=False as per paper
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False) # bias=False as per paper

        # If input and output channels are different, or stride is not 1,
        # we need a projection shortcut to match dimensions.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Apply shortcut if dimensions don't match
        out += self.shortcut(identity)
        # out = self.relu(out) # Final ReLU after adding the shortcut
        return out

class PinillaUnet_config(PretrainedConfig):
    def __init__(self, in_channels=3, out_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels


class PinillaUnet_arch(PreTrainedModel):
    """
    A U-Net based generator architecture as described in Figure 8.
    It has seven scales with six consecutive downsampling and upsampling operations.
    The number of channels are 64, 128, 256, 512.
    Four successive residual blocks are adopted in downscaling and upscaling.
    """
    config_class = PinillaUnet_config
    def __init__(self, config):
        super(PinillaUnet_arch, self).__init__(config)
        
        in_channels = config.in_channels
        out_channels = config.out_channels

        # Encoder path (Downscaling)
        # Scale 1
        self.enc1_conv = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False) # Downsampling from (H, W) to (H/2, W/2)
        self.enc1_res_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Scale 2
        self.enc2_conv = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc2_res_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        # Scale 3
        self.enc3_conv = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc3_res_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        # Scale 4 (bottleneck)
        self.enc4_conv = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc4_res_blocks = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        # Decoder path (Upscaling)
        # Scale 4 (Upsampled from Scale 4 bottleneck)
        self.dec4_tconv = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec4_res_blocks = nn.Sequential(
            ResidualBlock(256 + 256, 256), # Concatenation + current features
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        # Scale 3
        self.dec3_tconv = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec3_res_blocks = nn.Sequential(
            ResidualBlock(128 + 128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )

        # Scale 2
        self.dec2_tconv = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec2_res_blocks = nn.Sequential(
            ResidualBlock(64 + 64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.dec1_tconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec1_res_blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )

        # Final layer to map back to output channels
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1, bias=False) # Use 1x1 conv

    def forward(self, x):
        # Encoder
        # Scale 1
        enc1 = self.enc1_conv(x)
        enc1 = self.enc1_res_blocks(enc1)

        # Scale 2
        enc2 = self.enc2_conv(enc1)
        enc2 = self.enc2_res_blocks(enc2)

        # Scale 3
        enc3 = self.enc3_conv(enc2)
        enc3 = self.enc3_res_blocks(enc3)

        # Scale 4 (bottleneck)
        enc4 = self.enc4_conv(enc3)
        enc4 = self.enc4_res_blocks(enc4)

        # Decoder
        # Scale 4
        dec4 = self.dec4_tconv(enc4)
        # Skip connection from Scale 3 encoder features
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.dec4_res_blocks(dec4)

        # Scale 3
        dec3 = self.dec3_tconv(dec4)
        # Skip connection from Scale 2 encoder features
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3_res_blocks(dec3)

        # Scale 2
        dec2 = self.dec2_tconv(dec3)
        # Skip connection from Scale 1 encoder features
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.dec2_res_blocks(dec2)

        # Scale 1
        dec1 = self.dec1_tconv(dec2)
        dec1 = self.dec1_res_blocks(dec1)

        # Output layer
        out = self.out_conv(dec1)
        # breakpoint()
        # The paper suggests a specific output activation or no activation,
        # and then combined with the input noise.
        # For a generic reconstruction, sigmoid or tanh is common if target is normalized.
        # If reconstructing raw pixel values, linear might be appropriate.
        # Here, we'll output linear, and the loss function will handle range.
        return out

# Initialize Discriminator (example GAN structure, needs to be implemented fully)
# The paper mentions Disc and VGG16. VGG16 is used for perceptual loss.
# Disc would be a typical patch-GAN or image-level discriminator.
# For demonstration, let's create a placeholder Discriminator.
class PinillaDiscriminator_config(PretrainedConfig):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels

class PinillaDiscriminator_arch(PreTrainedModel):
    config_class = PinillaDiscriminator_config
    def __init__(self, config):
        super(PinillaDiscriminator_arch, self).__init__(config)
        in_channels = config.in_channels

        # Simplified discriminator structure
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0), # Output a single value per patch
            nn.Sigmoid() # Output probability
        )

    def forward(self, x):
        # For a patch-GAN, the output shape would be (batch_size, 1, H'/W')
        # Here we'll flatten it for BCE loss.
        output = self.conv(x)
        return output.view(output.size(0), -1) # Flatten to (batch_size, N)

class PerceptualLoss(nn.Module):
    """
    A simplified perceptual loss using VGG16 features.
    In a real implementation, you would load a pre-trained VGG16 and extract features
    from specific layers (e.g., relu1_2, relu2_2, relu3_3).
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Select feature extraction layers. Common choices are from 'relu' activations.
        # The specific layers might need tuning based on the paper's implementation.
        self.features = nn.Sequential(
            *list(vgg16.features.children())[:23] # Up to conv4_3 (relu4_3 is the 23rd layer)
        )
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, pred_img, target_img):
        # Ensure images are 3-channel and normalized if VGG was trained on ImageNet
        # (typically scaled to [0, 1] or [-1, 1] and then normalized with mean/std)
        # For simplicity here, we'll assume compatible input.
        # If your images are [0, 255], you'll need to normalize them similarly to VGG's training.

        pred_features = self.features(pred_img)
        target_features = self.features(target_img)

        # Calculate MSE loss between feature maps
        loss = self.mse_loss(pred_features, target_features)
        return loss

class CombinedLoss(nn.Module):
    """
    Combines Adversarial Loss, PSNR Loss, and Perceptual Loss, as described in Figure 8.
    L_total = Ï1 * L_adv + Ï2 * L_PSNR + Ï3 * L_Percep
    """
    def __init__(self, sigma1=1.0, sigma2=1.0, sigma3=1.0, perceptual_weight_layer_idx=None):
        super(CombinedLoss, self).__init__()
        self.sigma1 = sigma1 # Weight for Adversarial Loss
        self.sigma2 = sigma2 # Weight for PSNR Loss
        self.sigma3 = sigma3 # Weight for Perceptual Loss

        self.adversarial_loss = nn.BCELoss() # Assuming a binary GAN setup
        self.psnr_loss = None # Needs to be calculated
        self.perceptual_loss = PerceptualLoss()

    def calculate_psnr(self, pred_img, target_img, max_val=1.0):
        """Calculates Peak Signal-to-Noise Ratio (PSNR)."""
        mse = torch.mean((pred_img - target_img)**2)
        if mse == 0:
            return torch.tensor(100.0) # High PSNR if images are identical
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr

    def forward(self, pred_img, target_img, disc_real_output, disc_fake_output, generator_output_for_disc):
        # Adversarial Loss (for the generator)
        # Generator wants discriminator to classify fake images as real.
        # Typically uses BCE Loss with target labels of 1s.
        labels_real = torch.ones_like(disc_real_output)
        labels_fake = torch.zeros_like(disc_fake_output)
        loss_adv_real = self.adversarial_loss(disc_real_output, labels_real) # Discriminator classifying real
        loss_adv_fake = self.adversarial_loss(disc_fake_output, labels_fake) # Discriminator classifying fake
        # Generator aims to minimize loss_adv_fake (i.e., fool the discriminator)
        # We use loss_adv_fake as the L_adv for the generator's optimization
        loss_adv = loss_adv_fake

        # PSNR Loss
        # PSNR is maximized, so we minimize -PSNR, which is equivalent to minimizing MSE scaled inversely by PSNR.
        # A common practice is to use MSE directly and potentially scale its contribution.
        # If the paper directly uses PSNR in the loss, it might be 1/PSNR or similar.
        # For this implementation, let's calculate MSE and imply it's related to PSNR minimization.
        # If the goal is to maximize PSNR, you'd maximize a related metric (e.g. minimizing MSE).
        # L_PSNR ~ MSE
        mse_loss_fn = nn.MSELoss()
        loss_psnr = mse_loss_fn(pred_img, target_img) # This is the MSE component

        # Perceptual Loss
        loss_percep = self.perceptual_loss(pred_img, target_img)

        # Total Loss
        total_loss = (self.sigma1 * loss_adv) + \
                     (self.sigma2 * loss_psnr) + \
                     (self.sigma3 * loss_percep)

        return total_loss, loss_adv, loss_psnr, loss_percep

# --- Example Usage ---

if __name__ == "__main__":
    # Model parameters from paper description
    in_channels = 3 # Assuming RGB images
    out_channels = 3
    num_scales = 7 # Implied by downscaling/upscaling stages

    # Initialize Generator
    generator = PinillaUnet_arch(in_channels=in_channels, out_channels=out_channels)
    print("U-Net Generator Architecture:")
    # print(generator)

    

    discriminator = PinillaDiscriminator_arch(in_channels=out_channels)
    print("\nPlaceholder Discriminator Architecture:")
    # print(discriminator)

    # Initialize Loss function with weights from paper description
    # These weights (Ï1, Ï2, Ï3) are hyperparameters to be tuned.
    # The paper states: "weighted combination of PSNR between estimated and ground truth images,
    # LPSNR, and perceptual losses LAdv and LPercep, with weights Ï1, Ï2, and Ï3."
    # This phrasing is a bit ambiguous. If LAdv refers to adversarial loss, then the combination is:
    # L_total = Ï1 * L_adv + Ï2 * L_PSNR + Ï3 * L_Percep
    # If L_PSNR is indeed PSNR, and not MSE. Using MSE as a proxy for L_PSNR.
    # Let's assume L_PSNR ~ MSE.
    combined_loss_fn = CombinedLoss(sigma1=0.1, sigma2=1.0, sigma3=0.1) # Example weights

    # --- Create Dummy Input and Target Images ---
    batch_size = 2
    img_height = 256
    img_width = 256

    # Generate a dummy blurred input image (e.g., from noise + blurred image)
    # The paper mentions "Noise map Blurred per channel image" as input.
    # For demonstration, let's use a simple RGB image as input.
    # If the input is truly constructed from noise + blurred, you'd create that here.
    input_image = torch.randn(batch_size, in_channels, img_height, img_width)
    # If your model is trained on images in [0, 1] range
    # input_image = torch.rand(batch_size, in_channels, img_height, img_width)

    # Generate a dummy target ground truth image
    target_image = torch.randn(batch_size, out_channels, img_height, img_width)
    # If your model is trained on images in [0, 1] range
    # target_image = torch.rand(batch_size, out_channels, img_height, img_width)

    # --- Forward Pass ---
    # Generator produces a reconstruction
    reconstructed_image = generator(input_image)

    # Discriminator evaluates real (ground truth) and fake (reconstructed) images
    disc_real_output = discriminator(target_image)
    disc_fake_output = discriminator(reconstructed_image.detach()) # Detach from generator graph for discriminator training

    # Calculate the combined loss for the generator
    generator_loss, l_adv, l_psnr, l_percep = combined_loss_fn(
        reconstructed_image,
        target_image,
        discriminator(reconstructed_image), # Pass through discriminator again for generator loss
        disc_real_output, # Not directly used for generator's BCE loss, but needed in combined loss for total computation
        reconstructed_image # Passed for generator BCE loss calculation
    )

    print(f"\n--- Example Training Step ---")
    print(f"Input image shape: {input_image.shape}")
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    print(f"Target image shape: {target_image.shape}")
    print(f"Discriminator real output shape: {disc_real_output.shape}")
    print(f"Discriminator fake output shape: {disc_fake_output.shape}")
    print(f"Generator Loss: {generator_loss.item():.4f}")
    print(f"  - L_adv (Generator): {l_adv.item():.4f}")
    print(f"  - L_PSNR (proxy MSE): {l_psnr.item():.4f}")
    print(f"  - L_Percep: {l_percep.item():.4f}")

    # --- Optimizer Setup ---
    # You'd typically use Adam optimizer for GANs.
    # Learning rate, betas, etc., would be tuned.
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # --- Backpropagation (for one step) ---

    # Train Discriminator
    discriminator_optimizer.zero_grad()

    # Discriminator wants to classify real images as real (label=1) and fake images as fake (label=0).
    labels_real = torch.ones_like(disc_real_output)
    labels_fake = torch.zeros_like(disc_fake_output)

    # Loss for discriminator on real images
    loss_d_real = nn.BCELoss()(disc_real_output, labels_real)
    # Loss for discriminator on fake images
    loss_d_fake = nn.BCELoss()(disc_fake_output, labels_fake)

    # Total discriminator loss
    loss_d = (loss_d_real + loss_d_fake) * 0.5 # Average the two losses
    loss_d.backward()
    discriminator_optimizer.step()

    print(f"Discriminator Loss (Train step): {loss_d.item():.4f}")

    # Train Generator
    generator_optimizer.zero_grad()

    # Generator loss is calculated using the combined loss function.
    # We re-pass the reconstructed_image through the discriminator to calculate the generator's adversarial loss component.
    generator_loss.backward()
    generator_optimizer.step()

    print(f"Generator loss backpropagated and updated.")
