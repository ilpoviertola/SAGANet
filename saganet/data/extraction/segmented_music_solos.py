import csv
import logging
import typing as tp
from pathlib import Path
from math import ceil

import torch
from torchvision.transforms import v2
from torch.utils.data.dataset import Dataset
from torio.io import StreamingMediaDecoder

from saganet.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class SegmentedMusicSolos(Dataset):
    def __init__(
        self,
        root: tp.Union[str, Path],
        *,
        split: tp.Optional[str] = None,
        tsv_path: tp.Union[str, Path] = "sets/sms.tsv",
        mean_std_path: tp.Union[str, Path, None] = None,
        sample_rate: int = 44_100,
        duration_sec: float = 5.0,
        audio_samples: tp.Optional[int] = None,
        normalize_audio: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.tsv_path = Path(tsv_path)
        if mean_std_path is None:
            mean_std_path = self.root / "mean_std"
        self.mean_std_path = Path(mean_std_path)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.normalize_audio = normalize_audio
        self.audio_samples = self._get_audio_sample_amnt(audio_samples)
        self.clip_samples = int(_CLIP_FPS * self.duration_sec)
        self.sync_samples = int(_SYNC_FPS * self.duration_sec)
        self.split = split
        self.metadata = self._read_metadata()
        if local_rank == 0:
            log.info(f"URMP dataset loaded with {len(self.metadata)} samples")

        # data transforms
        self.clip_resize = v2.Resize(
            (_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC
        )
        self.clip_transform = v2.Compose(
            [
                self.clip_resize,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.sync_resize = v2.Resize(
            (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
        )
        self.sync_transform = v2.Compose(
            [
                self.sync_resize,
                v2.ToImage(),
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
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def _get_audio_sample_amnt(self, audio_samples: tp.Optional[int] = None) -> int:
        if audio_samples is None:
            audio_samples = int(self.sample_rate * self.duration_sec)
            # round to the next multiple of 1024
            audio_samples = ceil(audio_samples / 1024) * 1024
        else:
            audio_samples = int(audio_samples)
            effective_duration = audio_samples / self.sample_rate
            # make sure the duration is close enough, within 16ms
            assert (
                abs(effective_duration - self.duration_sec) < 0.016
            ), f"audio_samples {audio_samples} does not match duration_sec {self.duration_sec}"
        return audio_samples

    def _read_metadata(self) -> list[list[str]]:
        meta = []
        delimiter = "\t" if self.tsv_path.suffix == ".tsv" else ","
        with open(self.tsv_path, "r") as f:
            reader = csv.reader(f, delimiter=delimiter)
            next(reader)
            for row in reader:
                if self.split and row[6] != self.split:
                    continue
                if (self.root / row[7] / row[3] / f"{row[5]}.mp4").exists():
                    meta.append(row)
        return meta

    def _get_file_id(self, idx: int) -> Path:
        return (
            Path(self.metadata[idx][7]) / self.metadata[idx][3] / self.metadata[idx][5]
        )

    def _get_video_source_path(self, idx: int) -> Path:
        return (
            self.root
            / self.metadata[idx][7]
            / self.metadata[idx][3]
            / f"{self.metadata[idx][5]}.mp4"
        )

    def _get_mask_source_path(self, idx: int) -> Path:
        return (
            self.root
            / self.metadata[idx][7]
            / self.metadata[idx][3]
            / self.metadata[idx][5]
            / f"{self.metadata[idx][5]}_mask.mp4"
        )

    def _get_label(self, idx: int) -> str:
        return self.metadata[idx][4]

    def _read_video(self, path: Path) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format="rgb24",
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.add_basic_audio_stream(
            frames_per_chunk=self.audio_samples,
            sample_rate=self.sample_rate,
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = self.clip_transform(data_chunk[0])[: self.clip_samples]
        sync_chunk = self.sync_transform(data_chunk[1])[: self.sync_samples]
        audio_chunk = data_chunk[2]
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
        if audio_chunk.shape[0] < self.audio_samples:
            audio_chunk = torch.nn.functional.pad(
                audio_chunk, (0, self.audio_samples - audio_chunk.shape[0])
            )
        else:
            audio_chunk = audio_chunk[: self.audio_samples]

        return {
            "clip_video": clip_chunk,
            "sync_video": sync_chunk,
            "audio": audio_chunk,
        }

    def _read_mask_video(self, path: Path) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="gray",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        mask_chunk = self.mask_transform(data_chunk[0])[: self.sync_samples]
        return {"mask_video": mask_chunk}

    def _get_data_chunks(self, video: Path, mask: Path) -> dict[str, torch.Tensor]:
        data_chunks = {}
        data_chunks.update(self._read_video(video))
        data_chunks.update(self._read_mask_video(mask))
        return data_chunks

    def _check_chunks(
        self,
        data_chunks: tp.Dict[str, torch.Tensor],
    ):
        clip_chunk = data_chunks["clip_video"]
        sync_chunk = data_chunks["sync_video"]
        audio_chunk = data_chunks["audio"]
        mask_chunk = data_chunks["mask_video"]
        if clip_chunk is None:
            raise RuntimeError(f"CLIP video returned None")
        if clip_chunk.shape[0] < self.clip_samples:
            raise RuntimeError(
                f"CLIP video too short, expected {self.clip_samples}, got {clip_chunk.shape[0]}"
            )
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
        if audio_chunk is None:
            raise RuntimeError(f"Audio returned None")
        if audio_chunk.shape[0] < self.audio_samples:
            raise RuntimeError(
                f"Audio too short, expected {self.audio_samples}, got {audio_chunk.shape[0]}"
            )
        return True

    def sample(self, idx: int) -> dict[str, tp.Any]:
        video = self._get_video_source_path(idx)
        mask = self._get_mask_source_path(idx)
        label = self._get_label(idx)

        data_chunks = self._get_data_chunks(video, mask)
        self._check_chunks(data_chunks)

        data_chunks["id"] = self._get_file_id(idx).as_posix()  # type: ignore
        data_chunks["caption"] = label  # type: ignore
        data_chunks["name"] = video.name  # type: ignore
        return data_chunks

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tp.Optional[dict[str, tp.Any]]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading sample {self.metadata[idx][0]}: {e}")
            return None
