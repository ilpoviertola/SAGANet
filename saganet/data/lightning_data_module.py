import typing as tp

import lightning
from torch import Tensor
from torch.utils.data.dataloader import default_collate


class LightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.dataloader_kwargs = {
            "persistent_workers": False if num_workers == 0 else persistent_workers,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "batch_size": batch_size,
        }

    @staticmethod
    def collate(batch: tp.Any) -> tp.Any:
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    def get_normalization_stats(self) -> tp.Tuple[Tensor, Tensor]:
        """Get normalization statistics: mean and std.

        Returns:
            A tuple of (mean, std) tensors.
        """
        raise NotImplementedError
