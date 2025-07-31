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


class URMPDataset(Dataset):
    def __init__(
        self,
        root: tp.Union[str, Path],
        *,
        tsv_path: tp.Union[str, Path] = "sets/urmp.tsv",
        sample_rate: int = 44_100,
        duration_sec: float = 5.0,
        audio_samples: tp.Optional[int] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.tsv_path = Path(tsv_path)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.audio_samples = self._get_audio_sample_amnt(audio_samples)
        self.clip_samples = int(_CLIP_FPS * self.duration_sec)
        self.sync_samples = int(_SYNC_FPS * self.duration_sec)
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
        with open(self.tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if (self.root / f"{row[0]}.mp4").exists():
                    meta.append(row)
        return meta

    def _get_timestamps(self, idx: int) -> tuple[int, int]:
        start = self.metadata[idx][-2]
        end = self.metadata[idx][-1]
        return int(start), int(end)

    def _get_parent_dir(self, idx: int) -> str:
        return self._get_file_id(idx).split("/")[0]

    def _get_file_id(self, idx: int) -> str:
        return self.metadata[idx][1]

    def _get_song_num_and_name(self, idx: int) -> tuple[str, str]:
        pd = self._get_parent_dir(idx)
        pd_parts = pd.split("_")
        return pd_parts[0], pd_parts[1]

    def _get_instr_num_and_name(self, idx: int) -> tuple[str, str]:
        file_name = self._get_file_id(idx).split("/")[-1]
        fn_parts = file_name.split("_")
        return fn_parts[1], fn_parts[2]

    def _get_video_source_path(self, idx: int) -> Path:
        return self.root / f"{self.metadata[idx][0]}.mp4"

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
        return self.root / f"{self.metadata[idx][2]}.mp4"

    def _get_mask_detail_path(self, idx: int) -> Path:
        mpath = self._get_mask_source_path(idx)
        return mpath.parent / f"{mpath.stem}_detail.mp4"

    def _get_audio_source_path(self, idx: int) -> Path:
        return self.root / f"{self.metadata[idx][1]}.wav"

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
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = self.clip_transform(data_chunk[0])[: self.clip_samples]
        sync_chunk = self.sync_transform(data_chunk[1])[: self.sync_samples]
        return {"clip_video": clip_chunk, "sync_video": sync_chunk}

    def _read_detail_video(self, path: Path) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        sync_chunk = self.sync_transform(data_chunk[0])[: self.sync_samples]
        return {"sync_detail_video": sync_chunk}

    def _read_clip_video(self, path: Path) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        sync_chunk = self.clip_transform(data_chunk[0])[: self.clip_samples]
        return {"clip_video": sync_chunk}

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

    def _read_audio(self, path: Path) -> dict[str, torch.Tensor]:
        reader = StreamingMediaDecoder(path)
        reader.add_basic_audio_stream(
            frames_per_chunk=self.audio_samples,
            sample_rate=self.sample_rate,
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        audio_chunk = data_chunk[0]
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)
        if audio_chunk.shape[0] < self.audio_samples:
            audio_chunk = torch.nn.functional.pad(
                audio_chunk, (0, self.audio_samples - audio_chunk.shape[0])
            )
        else:
            audio_chunk = audio_chunk[: self.audio_samples]
        return {"audio": audio_chunk}

    def _get_data_chunks(
        self,
        video: Path,
        audio: Path,
        video_detail: Path,
        mask: Path,
        mask_detail: Path,
    ) -> dict[str, torch.Tensor]:
        data_chunks = {}
        data_chunks.update(self._read_video(video))
        data_chunks.update(self._read_clip_video(video_detail))
        data_chunks.update(self._read_detail_video(video_detail))
        data_chunks.update(self._read_mask_video(mask))
        data_chunks.update(self._read_mask_video(mask_detail, key="mask_detail_video"))
        data_chunks.update(self._read_audio(audio))
        return data_chunks

    def _check_chunks(
        self,
        data_chunks: tp.Dict[str, torch.Tensor],
    ):
        clip_chunk = data_chunks["clip_video"]
        sync_chunk = data_chunks["sync_video"]
        sync_detail_chunk = data_chunks["sync_detail_video"]
        # audio_chunk = data_chunks["audio"]
        mask_chunk = data_chunks["mask_video"]
        mask_detail_chunk = data_chunks["mask_detail_video"]
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
        # if audio_chunk is None:
        #     raise RuntimeError(f"Audio returned None")
        # if audio_chunk.shape[0] < self.audio_samples:
        #     raise RuntimeError(
        #         f"Audio too short, expected {self.audio_samples}, got {audio_chunk.shape[0]}"
        #     )
        return True

    def sample(self, idx: int) -> dict[str, tp.Any]:
        video = self._get_video_source_path(idx)
        video_detail = self._get_video_detail_path(idx)
        audio = self._get_audio_source_path(idx)
        mask = self._get_mask_source_path(idx)
        mask_detail = self._get_mask_detail_path(idx)
        label = self._get_label(idx)

        data_chunks = self._get_data_chunks(
            video, audio, video_detail, mask, mask_detail
        )
        self._check_chunks(data_chunks)

        data_chunks["id"] = self.metadata[idx][1]  # type: ignore
        data_chunks["mask_id"] = self.metadata[idx][2]  # type: ignore
        data_chunks["caption"] = label  # type: ignore
        data_chunks["name"] = self._get_audio_source_path(idx).stem  # type: ignore
        return data_chunks

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tp.Optional[dict[str, tp.Any]]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading sample {self.metadata[idx][0]}: {e}")
            return None
