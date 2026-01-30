import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function

from bubbleformer.layers import (
    HMLPEmbed, 
    HMLPDebed, 
    FiLMMLP,
    TransformerNeighborBlock
)
from bubbleformer.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["NeighborViT"]

@register_model("neighbor_vit")
class NeighborViT(nn.Module):
    def __init__(
        self,
        input_fields: int = 3,
        output_fields: int = 3,
        time_window: int = 12,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        processor_blocks: int = 12,
        num_fluid_params: int = 8,
    ):
        """
        Args:
            input_fields (int): Number of input fields
            output_fields (int): Number of output fields
            time_window (int): Number of time steps
            patch_size (int): Size of the square patch
            embed_dim (int): Dimension of the embedding
            num_heads (int): Number of attention heads
            processor_blocks (int): Number of processor blocks
            num_fluid_params (int): Number of fluid parameters for conditioning
        """
        super().__init__()
        self.embed = HMLPEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )

        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerNeighborBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            for _ in range(processor_blocks)
        ])

        self.debed = HMLPDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=embed_dim
        )
        
        self.sdf_proj = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1, dtype=torch.float32)
        self.temp_proj = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1, dtype=torch.float32)
        self.vel_proj = nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1, dtype=torch.float32)
        
    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        fluid_params: (B, num_fluid_params)
        """
        x = batch.input
        fluid_params = batch.fluid_params_tensor
        B, T, _, _, _ = x.shape
        
        input = x.clone()

        # Encode
        with record_function("encode"):
            x = rearrange(x, "b t c h w -> (b t) c h w")
            x = self.embed(x)
            x = rearrange(x, "(b t) c h w -> b t c h w", t=T)
            
        embed = x.clone()

        # Permute to better order for attention (B, T, H, W, C)
        # TODO: IDK if input should be in this format for the embedding or not...
        # I think conv's do support NHWC layout.
        x = rearrange(x, "b t c h w -> b t h w c").contiguous()

        # Apply FiLM conditioning on the embeddings
        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)

        # Attention blocks
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x = blk(x)

        x = rearrange(x, "b t h w c -> b t c h w").contiguous()
        
        # Skip connection from patch embeddings
        x = x + embed
       
        # Decode
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.debed(x)
        x = nn.functional.gelu(x)
        
        # convert to float32 for high-precision output projection
        x = x.to(torch.float32)
        
        # project to output fields
        sdf = self.sdf_proj(x)
        temp = self.temp_proj(x)
        vel = self.vel_proj(x)
        sdf = rearrange(sdf, "(b t) c h w -> b t c h w", b=B, t=T)
        temp = rearrange(temp, "(b t) c h w -> b t c h w", b=B, t=T)
        vel = rearrange(vel, "(b t) c h w -> b t c h w", b=B, t=T)
        x = torch.cat((sdf, temp, vel), dim=2)
        
        # Skip connection from the last timestep of the original input
        x = x + input[:, -1].unsqueeze(1).expand(-1, T, -1, -1, -1)
        
        return x