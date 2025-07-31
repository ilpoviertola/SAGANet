import csv
import logging
import typing as tp
from pathlib import Path
from math import ceil

import torch
from torchvision.transforms import v2
from torch.utils.data.dataset import Dataset
from torio.io import StreamingMediaDecoder
from tensordict import TensorDict
import numpy as np

from saganet.utils.dist_utils import local_rank

log = logging.getLogger()


_SYNC_SIZE = 224
_SYNC_FPS = 25.0
MAX_LOAD_ATTEMPTS = 10


class URMPDataset(Dataset):
    def __init__(
        self,
        root: tp.Union[str, Path],
        *,
        premade_mmap_dir: tp.Union[str, Path],
        data_dim: dict[str, int],
        tsv_path: tp.Union[str, Path] = "sets/urmp.tsv",
        duration_sec: float = 5.0,
    ):
        super().__init__()
        self.root = Path(root)
        self.data_dim = data_dim
        self.tsv_path = Path(tsv_path)
        self.duration_sec = duration_sec
        self.metadata = self._read_metadata()

        log.info(f"Loading precomputed mmap from {premade_mmap_dir}")
        premade_mmap_dir = Path(premade_mmap_dir)
        td = TensorDict.load_memmap(premade_mmap_dir)
        log.info(f"Loaded precomputed mmap from {premade_mmap_dir}")
        self.mean = td["mean"]
        self.std = td["std"]
        self.clip_features = td["clip_features"]
        self.text_features = td["text_features"]

        if local_rank == 0:
            log.info(f"Loaded {len(self)} samples.")
            log.info(f"Loaded mean: {self.mean.shape}.")
            log.info(f"Loaded std: {self.std.shape}.")
            log.info(f"Loaded clip_features: {self.clip_features.shape}.")
            log.info(f"Loaded text_features: {self.text_features.shape}.")

        assert (
            self.mean.shape[1] == self.data_dim["latent_seq_len"]
        ), f'{self.mean.shape[1]} != {self.data_dim["latent_seq_len"]}'
        assert (
            self.std.shape[1] == self.data_dim["latent_seq_len"]
        ), f'{self.std.shape[1]} != {self.data_dim["latent_seq_len"]}'

        assert (
            self.clip_features.shape[1] == self.data_dim["clip_seq_len"]
        ), f'{self.clip_features.shape[1]} != {self.data_dim["clip_seq_len"]}'
        assert (
            self.text_features.shape[1] == self.data_dim["text_seq_len"]
        ), f'{self.text_features.shape[1]} != {self.data_dim["text_seq_len"]}'

        assert (
            self.clip_features.shape[-1] == self.data_dim["clip_dim"]
        ), f'{self.clip_features.shape[-1]} != {self.data_dim["clip_dim"]}'
        assert (
            self.text_features.shape[-1] == self.data_dim["text_dim"]
        ), f'{self.text_features.shape[-1]} != {self.data_dim["text_dim"]}'

        self.video_exist = torch.tensor(1, dtype=torch.bool)
        self.text_exist = torch.tensor(1, dtype=torch.bool)

        self.sync_samples = int(_SYNC_FPS * self.duration_sec)
        self.sync_resize = v2.Resize(
            (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
        )
        self.sync_transform = v2.Compose(
            [
                self.sync_resize,
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mask_resize = v2.Resize(
            (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
        )
        self.mask_transform = v2.Compose(
            [
                self.mask_resize,
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def _read_metadata(self) -> list[list[str]]:
        meta = []
        with open(self.tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if (self.root / f"{row[0]}.wav").exists():
                    meta.append(row)
        return meta

    def _get_timestamps(self, idx: int) -> tuple[int, int]:
        file_id = self._get_file_id(idx)
        start = file_id.split("_")[-2]
        end = file_id.split("_")[-1]
        return int(start), int(end)

    def _get_parent_dir(self, idx: int) -> str:
        return self._get_file_id(idx).split("/")[0]

    def _get_file_id(self, idx: int) -> str:
        return self.metadata[idx][0]

    def _get_song_num_and_name(self, idx: int) -> tuple[str, str]:
        pd = self._get_parent_dir(idx)
        pd_parts = pd.split("_")
        return pd_parts[0], pd_parts[1]

    def _get_instr_num_and_name(self, idx: int) -> tuple[str, str]:
        file_name = self._get_file_id(idx).split("/")[-1]
        fn_parts = file_name.split("_")
        return fn_parts[1], fn_parts[2]

    def _get_video_source_path(self, idx: int) -> Path:
        start, end = self._get_timestamps(idx)
        pd = self._get_parent_dir(idx)
        return self.root / pd / f"Vid_{pd}_{start}_{end}.mp4"

    def _get_video_detail_path(self, idx: int) -> Path:
        start, end = self._get_timestamps(idx)
        pd = self._get_parent_dir(idx)
        snum, sname = self._get_song_num_and_name(idx)
        inum, iname = self._get_instr_num_and_name(idx)
        return (
            self.root
            / pd
            / f"Vid_{snum}_{sname}_{inum}_{iname}_{start}_{end}_detail.mp4"
        )

    def _get_mask_source_path(self, idx: int) -> Path:
        vdpath = self._get_video_detail_path(idx)
        fn_parts = vdpath.stem.split("_")
        fn = f"{fn_parts[3]}_{fn_parts[4]}_{fn_parts[1]}_{fn_parts[2]}_{fn_parts[5]}_{fn_parts[6]}"
        return vdpath.parent / f"Masks_{fn}.mp4"

    def _get_mask_detail_path(self, idx: int) -> Path:
        mpath = self._get_mask_source_path(idx)
        return mpath.parent / f"{mpath.stem}_detail.mp4"

    def _get_label(self, idx: int) -> str:
        return self.metadata[idx][1]

    def _read_video(
        self, path: Path, key: str = "sync_video"
    ) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        sync_chunk = self.sync_transform(data_chunk[0])[: self.sync_samples]
        return {key: sync_chunk}

    def _read_mask_video(
        self, path: Path, key: str = "mask_video"
    ) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="gray",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        mask_chunk = self.mask_transform(data_chunk[0])[: self.sync_samples]
        return {key: mask_chunk}

    def _get_data_chunks(
        self, video: Path, mask: Path, video_detail: Path, mask_detail: Path
    ) -> dict[str, torch.Tensor]:
        data_chunks = {}
        data_chunks.update(self._read_video(video))
        data_chunks.update(self._read_video(video_detail, "sync_detail_video"))
        data_chunks.update(self._read_mask_video(mask))
        data_chunks.update(self._read_mask_video(mask_detail, "mask_detail_video"))
        return data_chunks

    def _check_chunks(
        self,
        data_chunks: tp.Dict[str, torch.Tensor],
    ):
        sync_chunk = data_chunks["sync_video"]
        sync_detail_chunk = data_chunks["sync_detail_video"]
        mask_chunk = data_chunks["mask_video"]
        mask_detail_chunk = data_chunks["mask_detail_video"]
        if sync_chunk is None:
            raise RuntimeError(f"Sync video returned None")
        if sync_chunk.shape[0] < self.sync_samples:
            raise RuntimeError(
                f"Sync video too short, expected {self.sync_samples}, got {sync_chunk.shape[0]}"
            )
        if mask_chunk is None:
            raise RuntimeError(f"Mask video not found")
        if mask_chunk.shape[0] < self.sync_samples:
            raise RuntimeError(
                f"Mask video too short, expected {self.sync_samples}, got {mask_chunk.shape[0]}"
            )
        if sync_detail_chunk is None:
            raise RuntimeError(f"Sync detail video not found")
        if sync_detail_chunk.shape[0] < self.sync_samples:
            raise RuntimeError(
                f"Sync detail video too short, expected {self.sync_samples}, got {sync_detail_chunk.shape[0]}"
            )
        if mask_detail_chunk is None:
            raise RuntimeError(f"Mask detail video not found")
        if mask_detail_chunk.shape[0] < self.sync_samples:
            raise RuntimeError(
                f"Mask detail video too short, expected {self.sync_samples}, got {mask_detail_chunk.shape[0]}"
            )
        return True

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        latents: torch.Tensor = self.mean
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def get_memory_mapped_tensor(self) -> TensorDict:
        td = TensorDict(
            {
                "mean": self.mean,
                "std": self.std,
                "clip_features": self.clip_features,
                "text_features": self.text_features,
            }
        )
        return td

    def sample(self, idx: int) -> tuple[dict[str, tp.Any], bool]:
        video = self._get_video_source_path(idx)
        video_detail = self._get_video_detail_path(idx)
        mask = self._get_mask_source_path(idx)
        mask_detail = self._get_mask_detail_path(idx)
        label = self._get_label(idx)

        data_chunks = self._get_data_chunks(video, mask, video_detail, mask_detail)
        self._check_chunks(data_chunks)

        data_chunks["id"] = self._get_file_id(idx)  # type: ignore
        data_chunks["caption"] = label  # type: ignore
        data_chunks["name"] = video.name  # type: ignore
        return data_chunks, True

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tp.Optional[dict[str, tp.Any]]:
        sample_loaded, load_attempts = False, 0
        while not sample_loaded and load_attempts < MAX_LOAD_ATTEMPTS:
            try:
                data_chunk, sample_loaded = self.sample(idx)
            except Exception as e:
                log.error(f"Error loading sample {self._get_file_id(idx)}: {e}")
                sample_loaded = False
                idx = np.random.randint(0, len(self))
                load_attempts += 1

        data_chunk.update(
            {
                "clip_features": self.clip_features[idx],
                "text_features": self.text_features[idx],
                "video_exist": self.video_exist,
                "text_exist": self.text_exist,
                "a_mean": self.mean[idx],
                "a_std": self.std[idx],
            }
        )
        return data_chunk
