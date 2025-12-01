from typing import Tuple, Union
from pathlib import Path

from torch.utils.data import DataLoader
from .lightning_data_module import LightningDataModule

from .segmented_music_solos import SegmentedMusicSolos
from .urmp import URMPDataset
from saganet.model.sequence_config import CONFIG_44K_SA


class SMSDataModule(LightningDataModule):
    def __init__(
        self,
        sms_root: str,
        urmp_root: str,
        data_dim: dict[str, int],
        sms_train_mmap_dir: Union[str, None] = None,
        sms_val_mmap_dir: Union[str, None] = None,
        urmp_mmap_dir: Union[str, None] = None,
        sms_train_tsv_path: Union[str, None] = None,
        sms_val_tsv_path: Union[str, None] = None,
        urmp_tsv_path: Union[str, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dim = data_dim
        self.data_dim["latent_seq_len"] = CONFIG_44K_SA.latent_seq_len
        self.data_dim["clip_seq_len"] = CONFIG_44K_SA.clip_seq_len
        self.data_dim["sync_seq_len"] = CONFIG_44K_SA.sync_seq_len

        self.sms_root = sms_root
        assert Path(sms_root).exists(), f"{sms_root} does not exist"
        if sms_train_mmap_dir is None:
            sms_train_mmap_dir = f"{sms_root}/memmaps/sms-train"
        assert Path(sms_train_mmap_dir).exists(), f"{sms_train_mmap_dir} does not exist"
        self.sms_train_mmap_dir = sms_train_mmap_dir
        if sms_val_mmap_dir is None:
            sms_val_mmap_dir = f"{sms_root}/memmaps/sms-val"
        assert Path(sms_val_mmap_dir).exists(), f"{sms_val_mmap_dir} does not exist"
        self.sms_val_mmap_dir = sms_val_mmap_dir
        if sms_train_tsv_path is None:
            sms_train_tsv_path = f"{sms_root}/memmaps/sms-train.tsv"
        assert Path(sms_train_tsv_path).exists(), f"{sms_train_tsv_path} does not exist"
        self.sms_train_tsv_path = sms_train_tsv_path
        if sms_val_tsv_path is None:
            sms_val_tsv_path = f"{sms_root}/memmaps/sms-val.tsv"
        assert Path(sms_val_tsv_path).exists(), f"{sms_val_tsv_path} does not exist"
        self.sms_val_tsv_path = sms_val_tsv_path

        self.urmp_root = urmp_root
        assert Path(urmp_root).exists(), f"{urmp_root} does not exist"
        if urmp_mmap_dir is None:
            urmp_mmap_dir = f"{urmp_root}/memmaps/urmp-test"
        assert Path(urmp_mmap_dir).exists(), f"{urmp_mmap_dir} does not exist"
        self.urmp_mmap_dir = urmp_mmap_dir
        if urmp_tsv_path is None:
            urmp_tsv_path = f"{urmp_root}/memmaps/urmp-test.tsv"
        assert Path(urmp_tsv_path).exists(), f"{urmp_tsv_path} does not exist"
        self.urmp_tsv_path = urmp_tsv_path

    def get_normalization_stats(self) -> Tuple:
        if hasattr(self, "train_ds"):
            return self.train_ds.compute_latent_stats()
        elif hasattr(self, "test_ds"):
            return self.test_ds.compute_latent_stats()
        elif hasattr(self, "val_ds"):
            return self.val_ds.compute_latent_stats()
        else:
            raise ValueError("No dataset available to compute normalization stats.")

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = SegmentedMusicSolos(
                root=self.sms_root,
                split="train",
                premade_mmap_dir=self.sms_train_mmap_dir,
                data_dim=self.data_dim,
                tsv_path=self.sms_train_tsv_path,
            )
            self.val_ds = SegmentedMusicSolos(
                root=self.sms_root,
                split="val",
                premade_mmap_dir=self.sms_val_mmap_dir,
                data_dim=self.data_dim,
                tsv_path=self.sms_val_tsv_path,
            )
        elif stage == "validate":
            self.val_ds = SegmentedMusicSolos(
                root=self.sms_root,
                split="val",
                premade_mmap_dir=self.sms_val_mmap_dir,
                data_dim=self.data_dim,
                tsv_path=self.sms_val_tsv_path,
            )
        elif stage == "test" or stage == "predict":
            self.test_ds = URMPDataset(
                root=self.urmp_root,
                premade_mmap_dir=self.urmp_mmap_dir,
                data_dim=self.data_dim,
                tsv_path=self.urmp_tsv_path,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            **self.dataloader_kwargs,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self.dataloader_kwargs,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            **self.dataloader_kwargs,
            collate_fn=self.collate,
        )
