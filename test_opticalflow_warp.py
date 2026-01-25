import torch
import torch.nn.functional as F

class ContinuousWarper:
    def __init__(self, height, width, device='cuda'):
        self.height = height
        self.width = width
        self.device = device
        
        # 1. Initialize the Base Grid (Identity)
        # We keep this state to track the "total displacement" over time
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height, device=device),
                                        torch.arange(0, width, device=device),
                                        indexing='ij')
        
        # Stack to (H, W, 2)
        self.current_grid = torch.stack((grid_x, grid_y), dim=2).float()
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
        self.current_grid = self.current_grid - flow
        
        # 3. Normalize Grid to [-1, 1] for grid_sample
        # We create a temporary normalized grid for sampling
        norm_grid = self.current_grid.clone()
        norm_grid[..., 0] = 2.0 * norm_grid[..., 0] / max(self.width - 1, 1) - 1.0
        norm_grid[..., 1] = 2.0 * norm_grid[..., 1] / max(self.height - 1, 1) - 1.0
        
        # 4. Sample from the ORIGINAL image
        # Because we sample from original_image every time, we lose no quality.
        warped_frame = F.grid_sample(original_image, norm_grid, 
                                     mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return warped_frame

# --- Usage Example ---

# Setup
H, W = 256, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warper = ContinuousWarper(H, W, device)

# Create a dummy image (e.g., a circle)
img = torch.zeros(1, 3, H, W, device=device)
y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
mask = ((x - 128)**2 + (y - 128)**2) < 50**2
img[:, :, mask] = 1.0 

# Define a static flow field (e.g., everything moves right by 1 pixel per frame)
flow = torch.zeros(1, 2, H, W, device=device)
flow[:, 0, :, :] = 1.0 # u = 1 (right)

frames = []

# Generate 30 frames
for t in range(30):
    # Notice we pass 'img' (the original), not the output of the previous step
    new_frame = warper.step(img, flow)
    frames.append(new_frame)
    
    # In a real app, you might update 'flow' here if the wind changes direction

print(f"Generated {len(frames)} frames. The circle should move 30 pixels to the right without blurring.")