import pytest
import torch
from bubbleformer.models import get_model

@pytest.mark.parametrize("fields", [1, 2])
@pytest.mark.parametrize("patch_size", [4, 8, 16])
@pytest.mark.parametrize("embed_dim", [192, 384])
def test_avit(fields, patch_size, embed_dim):
    """
    Test AViT model with random configs
    """
    model_params = {
        "fields": fields,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_heads": 4,
        "processor_blocks": 4,
        "drop_path": 0.1
    }
    spatial_dims = 64, 64
    model = get_model("avit", **model_params)
    x = torch.randn(2, 3, fields, *spatial_dims) # (B, T, C, H, W)
    y = model(x) # (B, T, C, H, W)

    assert y.shape == (2, 3, fields, *spatial_dims)
