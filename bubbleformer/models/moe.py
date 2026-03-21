import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from einops import rearrange
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding

from bubbleformer.layers import (
    HMLPEmbed, 
    HMLPDebed,
    LinearEmbed,
    LinearDebed,
    FiLMMLP,
    TransformerMoEBlock,
    TransformerAxialMoEBlock,
    TransformerNeighborMoEBlock,
    TransformerSpatialNeighborMoEBlock
)
from bubbleformer.data.batching import CollatedBatch
from ._api import register_model

__all__ = ["ViTMoE", "AxialMoE", "NeighborMoE"]

class MoEBase(nn.Module):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__()
        self.embed = LinearEmbed(
            patch_size=patch_size,
            in_channels=input_fields,
            embed_dim=embed_dim,
        )
        
        self.film_embed = FiLMMLP(num_fluid_params, embed_dim)

        # Every attention block reuses the same frequencies, so we only need to compute them once.
        self.rotary_emb = RotaryEmbedding(
            # NOTE: This must be smaller than the head dim. 
            dim=(embed_dim // num_heads) // 3,
            freqs_for="pixel",
            max_freq=256,
            # We want a [Batch, Seq1, Seq2, Seq3, Heads, Dim] layout
            seq_before_head_dim=True
        )
        
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])

        self.out_norm = nn.RMSNorm(embed_dim)
        self.debed = LinearDebed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=output_fields
        )        
        
    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        """
        x: (B, T, H, W, C)
        fluid_params: (B, num_fluid_params)
        """
        x = batch.input
        fluid_params = batch.fluid_params_tensor(x.device)        
        input = x
        assert input.dtype == torch.float32
        assert fluid_params.dtype == torch.float32
        
        with record_function("encode"):
            x = self.embed(x)
        embed = x

        with record_function("film_embed"):
            x = self.film_embed(x, fluid_params)
        fluid_embed = x
        
        # Get axial frequencies for rotary embedding.
        # We expand the dims so that it matches [B, T, H, W, heads, head_dim] used in the attention layers.
        # These are unlearned, so do with no_grad.
        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = embed.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        # Attention blocks, tracking the MoE output for the routing losses
        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
               x, moe_output = blk(x, rotary_freqs)
               moe_outputs.append(moe_output)
        
        # Skip connections from patch and fluid embeddings
        x = x + embed + fluid_embed   

        with record_function("debed"):
            x = self.out_norm(x)
            x = self.debed(x)
        
        # Skip connection from the input
        x = x + input
        
        return x, moe_outputs
    
@register_model("vit_moe")
class ViTMoE(MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )

@register_model("axial_moe")
class AxialMoE(MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        self.blocks = nn.ModuleList([
            TransformerAxialMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])

@register_model("spatial_neighbor_moe")
class SpatialNeighborMoE(MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        self.blocks = nn.ModuleList([
            TransformerSpatialNeighborMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])
        
@register_model("neighbor_moe")
@torch.compile(fullgraph=True)
class NeighborMoE(MoEBase):
    def __init__(
        self,
        input_fields: int,
        output_fields: int,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        processor_blocks: int,
        num_fluid_params: int,
        num_experts: int,
        topk: int,
        load_balance_loss_weight: float,
    ):
        super().__init__(
            input_fields=input_fields,
            output_fields=output_fields,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            processor_blocks=processor_blocks,
            num_fluid_params=num_fluid_params,
            num_experts=num_experts,
            topk=topk,
            load_balance_loss_weight=load_balance_loss_weight,
        )
        self.blocks = nn.ModuleList([
            TransformerNeighborMoEBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                topk=topk,
                load_balance_loss_weight=load_balance_loss_weight,
            )
            for _ in range(processor_blocks)
        ])