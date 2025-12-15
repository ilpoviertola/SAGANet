import csv
import typing as tp
from pathlib import Path
from fractions import Fraction
import logging

import torch
from torchvision.transforms import v2
from torch.utils.data.dataset import Dataset
from tensordict import TensorDict
import numpy as np
from lightning_fabric.utilities import rank_zero_info

from saganet.utils.dist_utils import local_rank
from .av_utils import resample_video, read_video_to_frames_and_audio_with_av

_SYNC_SIZE = 224
_SYNC_FPS = 25.0
MAX_LOAD_ATTEMPTS = 10
logger = logging.getLogger(__name__)


class SegmentedMusicSolos(Dataset):
    def __init__(
        self,
        root: tp.Union[str, Path],
        split: str,
        *,
        premade_mmap_dir: tp.Union[str, Path],
        data_dim: dict[str, int],
        tsv_path: tp.Union[str, Path] = "sets/sms.csv",
        duration_sec: float = 5.0,
    ):
        super().__init__()
        self.root = Path(root)
        self.data_dim = data_dim
        self.tsv_path = Path(tsv_path)
        self.duration_sec = duration_sec
        self.split = split
        self.metadata = self._read_metadata()

        rank_zero_info(f"Loading precomputed mmap from {premade_mmap_dir}")
        premade_mmap_dir = Path(premade_mmap_dir)
        td = TensorDict.load_memmap(premade_mmap_dir)
        rank_zero_info(f"Loaded precomputed mmap from {premade_mmap_dir}")
        self.mean = td["mean"]
        self.std = td["std"]
        self.clip_features = td["clip_features"]
        self.text_features = td["text_features"]

        if local_rank == 0:
            rank_zero_info(f"Loaded {len(self)} samples.")
            rank_zero_info(f"Loaded mean: {self.mean.shape}.")
            rank_zero_info(f"Loaded std: {self.std.shape}.")
            rank_zero_info(f"Loaded clip_features: {self.clip_features.shape}.")
            rank_zero_info(f"Loaded text_features: {self.text_features.shape}.")

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
        delim = "\t" if self.tsv_path.suffix == ".tsv" else ","
        with open(self.tsv_path, "r") as f:
            reader = csv.reader(f, delimiter=delim)
            next(reader)
            for row in reader:
                if (self.root / f"{row[0]}.mp4").exists():
                    meta.append(row)
        return meta

    def _get_file_id(self, idx: int) -> str:
        return self.metadata[idx][0]

    def _get_video_source_path(self, idx: int) -> Path:
        return self.root / f"{self.metadata[idx][0]}.mp4"

    def _get_video_detail_path(self, idx: int) -> Path:
        vpath = self._get_video_source_path(idx)
        return vpath.parent / vpath.stem / f"{vpath.stem}_detail.mp4"

    def _get_mask_source_path(self, idx: int) -> Path:
        vpath = self._get_video_source_path(idx)
        return vpath.parent / vpath.stem / f"{vpath.stem}_mask.mp4"

    def _get_mask_detail_path(self, idx: int) -> Path:
        vpath = self._get_video_source_path(idx)
        return vpath.parent / vpath.stem / f"{vpath.stem}_mask_detail.mp4"

    def _get_label(self, idx: int) -> str:
        return self.metadata[idx][1]

    def _read_video(
        self, path: Path, key: str = "sync_video"
    ) -> dict[str, torch.Tensor]:
        video, _, meta = self.load_video_from_file(
            path, 0, self.duration_sec, to_rgb=True
        )
        assert video.numel() > 0, f"Video {path} is empty"
        if meta["video_fps"] != _SYNC_FPS:
            video = resample_video(
                video,
                original_fps=meta["video_fps"],
                target_fps=_SYNC_FPS,
            )
        sync_chunk = self.sync_transform(video)[: self.sync_samples]
        return {key: sync_chunk}

    def _read_mask_video(
        self, path: Path, key: str = "mask_video"
    ) -> dict[str, torch.Tensor]:
        video, _, meta = self.load_video_from_file(
            path, 0, self.duration_sec, to_rgb=False
        )
        assert video.numel() > 0, f"Mask video {path} is empty"
        if meta["video_fps"] != _SYNC_FPS:
            video = resample_video(
                video,
                original_fps=meta["video_fps"],
                target_fps=_SYNC_FPS,
            )
        mask_chunk = self.mask_transform(video)[: self.sync_samples]
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
                logger.error(f"Error loading sample {self._get_file_id(idx)}: {e}")
                sample_loaded = False
                idx = np.random.randint(0, len(self))
                load_attempts += 1
        if not sample_loaded:
            raise RuntimeError(
                f"Failed to load a sample after {MAX_LOAD_ATTEMPTS} attempts."
            )

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

    @staticmethod
    def load_video_from_file(
        file: Path,
        v_start_s: tp.Union[float, Fraction] = 0,
        v_end_s: tp.Optional[tp.Union[float, Fraction]] = None,
        a_start_s: tp.Union[float, Fraction] = 0,
        a_end_s: tp.Optional[tp.Union[float, Fraction]] = None,
        to_rgb: bool = True,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]:
        """Load a video from a file.

        Args:
            file (Path): Filepath to the video.
            v_start_s (tp.Union[float, Fraction], optional): Start point (in seconds) for the RGB stream. Defaults to 0.
            v_end_s (tp.Optional[tp.Union[float, Fraction]], optional): End point (in seconds) for the RGB stream. Defaults to None.
            a_start_s (tp.Union[float, Fraction], optional): Start point (in seconds) for the audio stream. Defaults to 0.
            a_end_s (tp.Optional[tp.Union[float, Fraction]], optional): End point (in seconds) for the audio stream. Defaults to None.
            to_rgb (bool, optional): Whether to convert video frames to RGB. Defaults to True.

        Raises:
            ValueError: If video start time is greater than video end time.
            ValueError: If audio start time is greater than audio end time.

        Returns:
            tp.Tuple[torch.Tensor, torch.Tensor, tp.Dict[str, tp.Any]]: RGB (THWC), audio and metadata.
        """
        if v_start_s is not None and a_start_s is None:
            a_start_s = v_start_s
        if a_start_s is not None and v_start_s is None:
            v_start_s = a_start_s
        if v_end_s is not None and a_end_s is None:
            a_end_s = v_end_s
        if a_end_s is not None and v_end_s is None:
            v_end_s = a_end_s

        if v_start_s and v_end_s and v_start_s >= v_end_s:
            raise ValueError(
                f"Invalid video start and end time: {v_start_s} >= {v_end_s}"
            )
        if a_start_s and a_end_s and a_start_s >= a_end_s:
            raise ValueError(
                f"Invalid audio start and end time: {a_start_s} >= {a_end_s}"
            )

        return read_video_to_frames_and_audio_with_av(
            file, v_start_s, v_end_s, a_start_s, a_end_s, to_rgb
        )
