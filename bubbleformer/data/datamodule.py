from typing import Optional, List

from torch.utils.data import DataLoader
from lightning import LightningDataModule
from bubbleformer.data.dataset import BubblemlForecast

class BubblemlForecastDataModule(LightningDataModule):
    def __init__(
        self,
        train_paths: Optional[List[str]] = None,
        valid_paths: Optional[List[str]] = None,
        test_paths: Optional[List[str]] = None,
        norm: str = "none",
        time_window: int = 16,
        future_window: int = 16,
        batch_size: int = 4,
        num_workers: int = 8,
    ):
        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.test_paths = test_paths
        self.norm = norm
        self.time_window = time_window
        self.future_window = future_window
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
        self,
        stage: str
    ):
        if stage == "fit":
            self.train = BubblemlForecast(
                filenames=self.train_paths,
                norm=self.norm,
                time_window=self.time_window,
                future_window=self.future_window,
            )
            self.valid = BubblemlForecast(
                filenames=self.valid_paths,
                norm=self.norm,
                time_window=self.time_window,
                future_window=self.future_window,
            )
        if stage == "test":
            self.test = BubblemlForecast(
                filenames=self.test_paths,
                norm=self.norm,
                time_window=self.time_window,
                future_window=self.future_window,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return  return DataLoader(
            self.test,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True
        )

