import torch
import torch.nn as nn
from models.vision_mamba import PureVisionMamba

class MambaDiffusionModel(nn.Module):
    def __init__(self, img_size=256, in_channels=2, out_channels=1, **kwargs):
        """
        A Diffusion model predicting noise in the mask space conditioned on a CT scan image.
        Instead of a standard UNet, we rely exclusively on a pure Vision Mamba backbone.
        in_channels = 2 (Noisy Mask + Conditioning CT Image)
        """
        super().__init__()
        self.noise_predictor = PureVisionMamba(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
        
    def forward(self, noisy_mask, condition_image, timesteps):
        """
        noisy_mask: Tensor of shape (B, 1, H, W) -> Added noise to true mask
        condition_image: Tensor of shape (B, 1, H, W) -> True CT Image (Lung Window normalized)
        timesteps: LongTensor (B,)
        """
        # Concatenate noisy mask and condition image along the channel dimension
        x = torch.cat([noisy_mask, condition_image], dim=1) # (B, 2, H, W)
        
        # Predict the noise applied to the mask at time `timesteps`
        predicted_noise = self.noise_predictor(x, timesteps)
        
        return predicted_noise
