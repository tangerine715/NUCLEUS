"""
This module contains dataset class for Time Series Forecasting
Classes:
    BubbleMLForecast: Dataset class for BubbleML dataset
Author: Sheikh Md Shakeel Hassan
"""
from typing import List, Optional, Tuple

import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset


class BubblemlForecast(Dataset):
    """
    Dataset class for time series forecasting on the BubbleML dataset
    """
    def __init__(
        self,
        filenames: List[str],
        fields: Optional[List[str]] = None,
        norm: str = "none",
        time_window: int = 16,
    ):
        super().__init__()
        self.filenames = filenames
        self.fields = fields if fields is not None else ["dfun", "temperature", "velx", "vely"]
        self.norm = norm
        self.time_window = time_window
        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []

        self.max_temps = [self.heater_temp(filename) for filename in filenames]
        self.min_temps = [
            58 if "saturated" in filename.lower() else 50 for filename in filenames
        ]
        for h5_file in self.data:
            self.num_trajs.append(1)
            self.traj_lens.append(h5_file[fields[0]].shape[0])

        self.num_fields = len(self.fields)
        self.diff_term = torch.zeros(self.num_fields)
        self.div_term = torch.ones(self.num_fields)

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            total_len += num_traj * (traj_len - 2 * self.time_window + 1)
        return total_len

    def normalize(
            self,
            diff_term: Optional[torch.tensor] = None,
            div_term: Optional[torch.tensor] = None,
        ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculate channel-wise normalization constants and store in a tensor of shape (C,)
        Open each File object in self.data['files'] and calculate the channelwise
        mean and std of the data
        """
        diff_terms = []
        div_terms = []
        if diff_term is None and div_term is None:
            for i, h5_file in enumerate(self.data):
                assert all([f in h5_file.keys() for f in self.fields]), "Invalid fields"
                data_fields = {}
                for field in self.fields:
                    if field == "temperature":
                        max_temp_diff = self.max_temps[i] - self.min_temps[i]
                        data_fields[field] = h5_file[field][...] * max_temp_diff + self.min_temps[i]
                    else:
                        data_fields[field] = h5_file[field][...]

                if self.norm == "std":
                    diff_terms.append(
                        torch.tensor([data_fields[f].mean() for f in self.fields])
                    )
                    div_terms.append(
                        torch.tensor([data_fields[f].std() for f in self.fields])
                    )
                elif self.norm == "minmax":
                    diff_terms.append(
                        torch.tensor([data_fields[f].min() for f in self.fields])
                    )
                    div_terms.append(
                        torch.tensor([
                            data_fields[f].max() - data_fields[f].min() for f in self.fields
                        ])
                    )
                elif self.norm == "tanh":
                    diff_terms.append(
                        torch.tensor([
                            (data_fields[f].max() + data_fields[f].min()) / 2.0 for f in self.fields
                        ])
                    )
                    div_terms.append(
                        torch.tensor([
                            (data_fields[f].max() - data_fields[f].min()) / 2.0 for f in self.fields
                        ])
                    )
                else:
                    self.norm = "none"
                    return self.diff_term, self.div_term
            diff_term = torch.stack(diff_terms).mean(dim=0)
            div_term = torch.stack(div_terms).mean(dim=0) + 1e-8

        self.diff_term = diff_term
        self.div_term = div_term

        return self.diff_term, self.div_term

    def heater_temp(self, filename):
        """
        Extract the heater temperature from the filename
        """
        if "Twall" in filename:
            return int(filename.split("-")[-1].split(".")[0])
        else:
            return 103

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_per_traj = [
            x * (y - 2 * self.time_window + 1)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        temp_diff = self.max_temps[file_idx] - self.min_temps[file_idx]

        inp_slice = slice(start, start + self.time_window)
        out_slice = slice(start + self.time_window, start + 2 * self.time_window)

        inp_data = []
        out_data = []

        for field in self.fields:
            if field == "temperature":
                inp_data.append(
                    torch.tensor(
                        self.data[file_idx][field][inp_slice] * temp_diff + self.min_temps[file_idx]
                    )
                )
                out_data.append(
                    torch.tensor(
                        self.data[file_idx][field][out_slice] * temp_diff + self.min_temps[file_idx]
                    )
                )
            else:
                inp_data.append(torch.tensor(self.data[file_idx][field][inp_slice]))
                out_data.append(torch.tensor(self.data[file_idx][field][out_slice]))

        inp_data = torch.stack(inp_data)                                   # (C, T, H, W)
        out_data = torch.stack(out_data)                                   # (C, T, H, W)

        inp_data = (inp_data - self.diff_term.view(-1, 1, 1, 1)) / self.div_term.view(-1, 1, 1, 1)
        out_data = (out_data - self.diff_term.view(-1, 1, 1, 1)) / self.div_term.view(-1, 1, 1, 1)

        return inp_data.permute(1, 0, 2, 3), out_data.permute(1, 0, 2, 3)  # (T, C, H, W)
