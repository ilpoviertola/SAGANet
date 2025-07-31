import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from saganet.data.eval.audiocaps import AudioCapsData
from saganet.data.eval.video_dataset import MovieGen, VGGSound
from saganet.data.raw_music import MusicRaw
from saganet.data.segmented_music_solos import SegmentedMusicSolos
from saganet.data.urmp import URMPDataset
from saganet.data.eval.urmp import URMPDataset as URMPDatasetEval
from saganet.utils.dist_utils import local_rank

log = logging.getLogger()


# Re-seed randomness every time we start a worker
def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    log.debug(
        f"Worker {worker_id} re-seeded with seed {worker_seed} in rank {local_rank}"
    )


def load_raw_music_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    dataset = MusicRaw(
        root=data_cfg.root,
        tsv_path=data_cfg.tsv,
        split=data_cfg.split,
    )
    return dataset


def load_sms_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    dataset = SegmentedMusicSolos(
        root=data_cfg.root,
        split=data_cfg.split,
        premade_mmap_dir=data_cfg.memmap_dir,
        data_dim=cfg.data_dim,
        tsv_path=data_cfg.tsv,
    )
    return dataset


def load_urmp_data(cfg: DictConfig, data_cfg: DictConfig) -> Dataset:
    dataset = URMPDataset(
        root=data_cfg.root,
        premade_mmap_dir=data_cfg.memmap_dir,
        data_dim=cfg.data_dim,
        tsv_path=data_cfg.tsv,
    )
    return dataset


def setup_training_datasets(
    cfg: DictConfig,
) -> tuple[Dataset, DistributedSampler, DataLoader]:
    if cfg.get("raw_music_train", False):
        dataset = load_raw_music_data(cfg, cfg.data.RawMusic)  # type: ignore
    elif cfg.get("sms_train", False):
        dataset = load_sms_data(cfg, cfg.data.SMS)  # type: ignore
    else:
        raise NotImplementedError("Training dataset not implemented")

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    return dataset, sampler, loader


def setup_test_datasets(cfg):
    if cfg.get("raw_music_train", False):
        dataset = load_raw_music_data(cfg, cfg.data.RawMusic_test)
    elif cfg.get("sms_train", False):
        dataset = load_urmp_data(cfg, cfg.data.URMP)
    else:
        raise NotImplementedError("Test dataset not implemented")

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    sampler, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return dataset, sampler, loader


def setup_val_datasets(cfg: DictConfig) -> tuple[Dataset, DataLoader, DataLoader]:
    if cfg.get("raw_music_train", False):
        dataset = load_raw_music_data(cfg, cfg.data.RawMusic_val)
    elif cfg.get("sms_train", False):
        dataset = load_sms_data(cfg, cfg.data.SMS_val)
    else:
        raise NotImplementedError("Validation dataset not implemented")

    val_batch_size = cfg.batch_size
    val_eval_batch_size = cfg.eval_batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, val_loader = construct_loader(
        dataset,
        val_batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )
    _, eval_loader = construct_loader(
        dataset,
        val_eval_batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return dataset, val_loader, eval_loader


def setup_eval_dataset(
    dataset_name: str, cfg: DictConfig
) -> tuple[Dataset, DataLoader]:
    if dataset_name.startswith("audiocaps_full"):
        dataset = AudioCapsData(
            cfg.eval_data.AudioCaps_full.audio_path,
            cfg.eval_data.AudioCaps_full.csv_path,
        )
    elif dataset_name.startswith("audiocaps"):
        dataset = AudioCapsData(
            cfg.eval_data.AudioCaps.audio_path, cfg.eval_data.AudioCaps.csv_path
        )
    elif dataset_name.startswith("moviegen"):
        dataset = MovieGen(  # type: ignore
            cfg.eval_data.MovieGen.video_path,
            cfg.eval_data.MovieGen.jsonl_path,
            duration_sec=cfg.duration_s,
        )
    elif dataset_name.startswith("vggsound"):
        dataset = VGGSound(  # type: ignore
            cfg.eval_data.VGGSound.video_path,
            cfg.eval_data.VGGSound.csv_path,
            duration_sec=cfg.duration_s,
        )
    elif dataset_name.startswith("music"):
        dataset = MusicRaw(  # type: ignore
            root=cfg.eval_data.RawMusic.root,
            tsv_path=cfg.eval_data.RawMusic.tsv,
            split=cfg.eval_data.RawMusic.split,
        )
    elif dataset_name.startswith("urmp"):
        dataset = URMPDatasetEval(  # type: ignore
            root=cfg.eval_data.URMP.root,
            tsv_path=cfg.eval_data.URMP.tsv,
            duration_sec=cfg.duration_s,
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    pin_memory = cfg.pin_memory
    _, loader = construct_loader(
        dataset,
        batch_size,
        num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        error_avoidance=True,
    )
    return dataset, loader


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def construct_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
    error_avoidance: bool = True,
) -> tuple[DistributedSampler, DataLoader]:
    train_sampler: DistributedSampler = DistributedSampler(
        dataset, rank=local_rank, shuffle=shuffle
    )
    train_loader = DataLoader(
        dataset,
        batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=error_avoidance_collate if error_avoidance else None,
    )
    return train_sampler, train_loader
