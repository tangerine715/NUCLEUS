import dataclasses
import torch
from typing import Dict, List, Optional

@dataclasses.dataclass
class Data:
    input: torch.Tensor
    target: torch.Tensor
    fluid_params_tensor: torch.Tensor
    fluid_params_dict: Dict
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    rollout_steps: Optional[int] = None

@dataclasses.dataclass
class CollatedBatch:
    input: torch.Tensor
    target: Optional[torch.Tensor]
    fluid_params_tensor: torch.Tensor
    fluid_params_dict: List[Dict]
    x_grid: torch.Tensor
    y_grid: torch.Tensor
    dx: torch.Tensor
    dy: torch.Tensor
    rollout_steps: Optional[torch.Tensor] = None
    
    def get_input(self):
        r"""
        This returns a copy of self, but without the target data,
        so this can be directly passed into the model as an input.
        """
        return CollatedBatch(
            input=self.input,
            target=None,
            fluid_params_tensor=self.fluid_params_tensor,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def fliplr(self):
        return CollatedBatch(
            input=torch.fliplr(self.input),
            target=torch.fliplr(self.target),
            fluid_params_tensor=self.fluid_params_tensor,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
        
    def gaussian_noise(self, scale: float):
        return CollatedBatch(
            input=self.input + torch.normal(0, scale, self.input.shape, device=self.input.device),
            target=self.target,
            fluid_params_tensor=self.fluid_params_tensor,
            fluid_params_dict=self.fluid_params_dict,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            dx=self.dx,
            dy=self.dy,
            rollout_steps=self.rollout_steps
        )
    
def make_data(input, target, fluid_params_tensor, fluid_params_dict, downsample_factor: int, rollout_steps: Optional[int] = None):
    dx = (fluid_params_dict["x_max"] - fluid_params_dict["x_min"]) / (fluid_params_dict["num_blocks_x"] * int(fluid_params_dict["nx_block"]))
    dy = (fluid_params_dict["y_max"] - fluid_params_dict["y_min"]) / (fluid_params_dict["num_blocks_y"] * int(fluid_params_dict["ny_block"]))

    if downsample_factor > 1:
        dx *= downsample_factor
        dy *= downsample_factor

    # + dx / 2 since we're using a cell-centered grid.
    x_grid = torch.arange(fluid_params_dict["x_min"], fluid_params_dict["x_max"], dx) + dx / 2
    y_grid = torch.arange(fluid_params_dict["y_min"], fluid_params_dict["y_max"], dy) + dy / 2

    return Data(
        input=input,
        target=target,
        fluid_params_tensor=fluid_params_tensor,
        fluid_params_dict=fluid_params_dict,
        x_grid=x_grid,
        y_grid=y_grid,
        dx=dx,
        dy=dy,
        rollout_steps=rollout_steps
    )

def collate(data: List[Data]):    
    return CollatedBatch(
        input=torch.stack([d.input for d in data]),
        target=torch.stack([d.target for d in data]),
        fluid_params_tensor=torch.stack([d.fluid_params_tensor for d in data]),
        fluid_params_dict=[d.fluid_params_dict for d in data],
        x_grid=torch.stack([d.x_grid for d in data]),
        y_grid=torch.stack([d.y_grid for d in data]),
        dx=torch.tensor([d.dx for d in data]),
        dy=torch.tensor([d.dy for d in data]),
    )