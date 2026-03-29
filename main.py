import argparse
from training.train import train_mamba_diffusion
from inference.predict import predict_diffusion_mask
import os
import numpy as np
import matplotlib.pyplot as plt

def setup_pylidc_config(manifest_dir):
    """Seamlessly generate the .pylidcrc to prevent dependency crashes."""
    pylidc_conf_path = os.path.expanduser('~/.pylidcrc')
    if not os.path.exists(pylidc_conf_path):
        abs_path = os.path.abspath(os.path.join(manifest_dir, "LIDC-IDRI"))
        with open(pylidc_conf_path, 'w') as f:
            f.write(f"[dicom]\npath = {abs_path}\nwarn = True\n")
        print(f"Auto-generated global pylidc config at: {pylidc_conf_path}")

def main():
    parser = argparse.ArgumentParser(description="Vision Mamba Diffusion Pipeline for Lung Nodule Segmentation")
    parser.add_argument('--train', action='store_true', help="Run the training loop")
    parser.add_argument('--predict', action='store_true', help="Run inference to generate a nodule mask")
    parser.add_argument('--manifest_dir', type=str, default="manifest-1585167679499", help="Path to manifest dataset directory containing LIDC-IDRI")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training (keep small for Mamba GPU mem)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Ensure pylidc is configured seamlessly for the user
    setup_pylidc_config(args.manifest_dir)
    
    if args.train:
        print(f"[*] Starting Training for {args.epochs} epochs with batch size {args.batch_size}...")
        train_mamba_diffusion(manifest_dir=args.manifest_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print("[*] Training Routine Completed!")
        
    elif args.predict:
        print("[*] Starting Inference Mask Generation via Pure Vision Mamba Denoising...")
        ct_slice, mask = predict_diffusion_mask(manifest_dir=args.manifest_dir, wait_steps=100)
        
        # Save the visualization
        os.makedirs('docs', exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(ct_slice, cmap='gray')
        axes[0].set_title("Original CT Slice")
        axes[0].axis('off')
        
        axes[1].imshow(ct_slice, cmap='gray')
        axes[1].imshow(mask, cmap='hot', alpha=0.5)
        axes[1].set_title("Generated Mask Overlay")
        axes[1].axis('off')
        
        binary_mask = (mask > 0.5).astype(np.float32)
        preserved_region = ct_slice * binary_mask
        axes[2].imshow(preserved_region, cmap='gray')
        axes[2].set_title("Preserved Target Region")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('docs/sample_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("[*] Prediction completed and saved to docs/sample_prediction.png successfully!")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
