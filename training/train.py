import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from models.diffusion_mamba import MambaDiffusionModel
from data.dataset import LIDCIDRIDataset
from tqdm import tqdm

def train_mamba_diffusion(manifest_dir='manifest-1585167679499', epochs=100, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    dataset = LIDCIDRIDataset(manifest_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the Mamba-based Diffusion Model
    model = MambaDiffusionModel(img_size=256, in_channels=2, out_channels=1).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Standard DDPM Scheduler from Huggingface diffusers
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    
    model.train()
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in loop:
                condition_image = batch['image'].to(device) # Original CT Scan
                clean_mask = batch['mask'].to(device)       # Ground Truth Node Mask
                B = clean_mask.shape[0]

                # Sample random noise to add to the masks
                noise = torch.randn_like(clean_mask)
                
                # Sample a random timestep for each image in the batch
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                
                # Add noise to the clean masks according to the noise magnitude at each timestep (Forward Diffusion Process)
                noisy_mask = noise_scheduler.add_noise(clean_mask, noise, timesteps)
                
                optimizer.zero_grad()
                
                # Predict the noise using Vision Mamba model
                noise_pred = model(noisy_mask, condition_image, timesteps)
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), "mamba_diffusion_lung_nodule.pth")
                print(f"  [Checkpoint saved at epoch {epoch+1}]")
                
    except KeyboardInterrupt:
        print(f"\n[!] Training interrupted! Saving current model weights...")
        torch.save(model.state_dict(), "mamba_diffusion_lung_nodule.pth")
        print(f"[OK] Model saved to mamba_diffusion_lung_nodule.pth")
        return
        
    torch.save(model.state_dict(), "mamba_diffusion_lung_nodule.pth")
    print(f"[OK] Final model saved to mamba_diffusion_lung_nodule.pth")

if __name__ == '__main__':
    train_mamba_diffusion()
