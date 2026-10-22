import pytest
import torch
from PIL import Image

from nucleus.data.layout import convert_layout
from nucleus.plot.diffusion_gif import save_diffusion_gif


@pytest.mark.parametrize('layout', ['t h w c', 't c h w'])
def test_comparison_gif_samples_multiple_windows(tmp_path, layout):
    window_paths = []
    for window_index in range(2):
        sdf = torch.arange(24).reshape(2, 3, 4).float() - 12 + window_index
        fields = sdf[..., None].expand(-1, -1, -1, 4)
        window_path = tmp_path / f'window_{window_index}.pt'
        torch.save({
            'preds': convert_layout(fields, layout),
            'targets': convert_layout(fields, layout),
            'diffusion_raw_sdf': sdf,
            'surrogate_sdf': sdf,
            'surrogate_reinitialized_sdf': sdf,
        }, window_path)
        window_paths.append(window_path)
    result = save_diffusion_gif(window_paths, layout, 0, tmp_path / 'comparison.gif', max_frames=3)
    with Image.open(result) as animation:
        assert animation.format == 'GIF'
        assert animation.n_frames == 3
        assert animation.info['loop'] == 0
