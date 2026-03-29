import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.linear(x)

class VisionMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (B, L, D)
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual

class PureVisionMamba(nn.Module):
    def __init__(self, img_size=256, in_channels=2, out_channels=1, patch_size=16, embed_dim=256, depth=8):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # Time embedding for Diffusion Models
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Mamba Blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(d_model=embed_dim) for _ in range(depth)
        ])
        
        # Decoder (to reconstruct standard image size from patches)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 8, out_channels, kernel_size=2, stride=2)
        )
        
    def forward(self, x, t):
        # x represents concatenated [noisy_mask, cond_image] with shape (B, 2, H, W)
        B, C, H, W = x.shape
        
        # Time step embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # (B, embed_dim)
        
        # Extract patches
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        Hp, Wp = x.shape[2], x.shape[3]
        
        # Flatten and transpose for Mamba (Sequence processing)
        x = x.flatten(2).transpose(1, 2)  # (B, L, embed_dim)
        
        x = x + self.pos_embed # Add positional embedding
        x = x + t_emb.unsqueeze(1) # Add time embedding to all sequence tokens
        
        # Pass through Mamba Blocks
        for block in self.blocks:
            x = block(x)
            
        # Reshape back to 2D
        x = x.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        
        # Decode back to mask space
        return self.decoder(x)
