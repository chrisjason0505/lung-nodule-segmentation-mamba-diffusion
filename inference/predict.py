import torch
import numpy as np
import matplotlib.pyplot as plt
from models.diffusion_mamba import MambaDiffusionModel
from diffusers import DDPMScheduler
from data.dataset import LIDCIDRIDataset
import os

def predict_diffusion_mask(manifest_dir='manifest-1585167679499', wait_steps=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = LIDCIDRIDataset(manifest_dir)
    sample = dataset[0]
    condition_img = sample['image'].unsqueeze(0).to(device)

    # Initialize Model and load weights
    model = MambaDiffusionModel(img_size=256, in_channels=2, out_channels=1).to(device)
    weight_path = "mamba_diffusion_lung_nodule.pth"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"[OK] Loaded trained weights from {weight_path}")
    else:
        print("[!] No trained weights found, using untrained model")
    model.eval()

    # Denoising Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=wait_steps, beta_schedule='squaredcos_cap_v2')

    # Start with pure Gaussian Noise for the mask
    x_t = torch.randn_like(condition_img)

    # Denoise step by step
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            t_batch = torch.tensor([t], device=device).long()
            
            # Predict noise using the condition CT image and the current noisy mask
            noise_pred = model(x_t, condition_img, t_batch)
            
            # Compute previous image x_{t-1}
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample
            
    # The final output is the generated mask
    generated_mask = torch.sigmoid(x_t) # Assuming semantic segmentation mask in [0, 1] bounds
    
    return condition_img.cpu().numpy()[0, 0], generated_mask.cpu().numpy()[0, 0]

if __name__ == '__main__':
    ct, mask = predict_diffusion_mask()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(ct, cmap='gray')
    axes[0].set_title("Original CT Slice")
    
    axes[1].imshow(ct, cmap='gray')
    axes[1].imshow(mask, cmap='hot', alpha=0.5)
    axes[1].set_title("Generated Mask Overlay")
    
    binary_mask = (mask > 0.5).astype(np.float32)
    preserved_region = ct * binary_mask
    axes[2].imshow(preserved_region, cmap='gray')
    axes[2].set_title("Preserved Target Region")
    
    plt.tight_layout()
    plt.savefig('docs/sample_prediction.png')
