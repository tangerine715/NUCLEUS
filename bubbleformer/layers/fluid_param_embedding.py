import torch
import torch.nn as nn
from bubbleformer.layers.mlp import FiLMMLP
from typing import List, Dict

class FluidParamEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.fluid_params = [
            "inv_reynolds",
            "cpgas",
            "mugas",
            "rhogas",
            "thcogas",
            "stefan",
            "prandtl",
            "gravy",
            "bulk_temp",
            "heater_wallTemp",
            "heater_nucWaitTime",
            "heater_rcdAngle",
            "heater_advAngle",
            "heater_velContact",
            "heater_xMin",
            "heater_xMax",
        ]
        
        # Each fluid param has a unique base embedding vector
        self.base_embedding = nn.Embedding(len(self.fluid_params), embed_dim)

        # Based on the value of the fluid param, we modulate its base embedding.
        self.film_mlp = FiLMMLP(1, embed_dim)

    def unnest_dict(self, fluid_params_dict: Dict):
        unnested_dict = {}
        
        for key, value in fluid_params_dict.items():
            if isinstance(value, dict):
                unnested_value: Dict[str, ...] = self.unnest_dict(value)
                for unnested_key, unnested_value in unnested_value.items():
                    unnested_dict[f"{key}_{unnested_key}"] = unnested_value
            else:
                unnested_dict[key] = value
                
        return unnested_dict
        
    def _get_valid_params(self, fluid_params_dicts: List[Dict[str, ...]]):
        unnested = [self.unnest_dict(fluid_params_dict) for fluid_params_dict in fluid_params_dicts]
        return [
            dict([(k, v) for k, v in fluid_params_dict.items() if k in self.fluid_params]) for fluid_params_dict in unnested
        ]

    def forward(self, fluid_params_dicts: List[Dict[str, ...]]):
        fluid_params_dicts = self._get_valid_params(fluid_params_dicts)
        # All of the modulated embeddings are summed together to get the parameter embedding.
        param_embedding = torch.zeros(len(fluid_params_dicts), self.embed_dim, dtype=self.base_embedding.weight.dtype, device=self.base_embedding.weight.device)
        for batch_idx, fluid_params_dict in enumerate(fluid_params_dicts):

            param_indices = [self.fluid_params.index(param_name) for param_name in fluid_params_dict.keys()]
            param_values = [fluid_params_dict[param_name] for param_name in fluid_params_dict.keys()]
            
            param_indices = torch.tensor(param_indices, dtype=torch.long, device=self.base_embedding.weight.device)
            param_values = torch.tensor(param_values, dtype=self.base_embedding.weight.dtype, device=self.base_embedding.weight.device)
            base_embedding = self.base_embedding(param_indices)
            film_embedding = self.film_mlp(base_embedding[:, None, None, None, :], param_values.unsqueeze(-1)).squeeze()
            param_embedding[batch_idx] = film_embedding.sum(0)

        return param_embedding