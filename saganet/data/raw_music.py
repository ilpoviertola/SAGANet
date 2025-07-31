import logging
import typing as tp
from pathlib import Path
from math import ceil
from random import choice

import pandas as pd
import torch
import torchaudio
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
from torio.io import StreamingMediaDecoder
from tensordict import TensorDict

from saganet.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

STITCH_TRANSFORM_WARMUP = 100


# transforms
class RandomHorizontalFlip(v2.RandomHorizontalFlip):
    def __init__(
        self,
        p: float = 0.5,
        apply_to: tp.List[str] = ["clip_video", "sync_video", "mask_video"],
    ):
        super().__init__()
        self.p = p
        self.apply_to = apply_to

    def forward(self, x: tp.Dict[str, torch.Tensor]) -> tp.Dict[str, torch.Tensor]:
        if torch.rand(1) < self.p:
            for key in x:
                if key in self.apply_to:
                    x[key] = super().forward(x[key])
        return x


class RandomVerticalFlip(v2.RandomVerticalFlip):
    def __init__(
        self,
        p: float = 0.5,
        apply_to: tp.List[str] = ["clip_video", "sync_video", "mask_video"],
    ):
        super().__init__()
        self.p = p
        self.apply_to = apply_to

    def forward(self, x: tp.Dict[str, torch.Tensor]) -> tp.Dict[str, torch.Tensor]:
        if torch.rand(1) < self.p:
            for key in x:
                if key in self.apply_to:
                    x[key] = super().forward(x[key])
        return x


class ColorJitter(torch.nn.Module):
    def __init__(
        self,
        s: float = 1.0,
        apply_to: tp.List[str] = ["clip_video", "sync_video"],
    ) -> None:
        super().__init__()
        self.s = s
        # SimCLR params
        self.transform = v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.apply_to = apply_to

    def forward(self, item):
        for key in item:
            if key in self.apply_to:
                for i in range(item[key].size(0)):
                    item[key][i] = self.transform(item[key][i])
        return item


class IdentityTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, item):
        return item


class MusicRaw(Dataset):
    def __init__(
        self,
        root: tp.Union[str, Path],
        split: str,
        *,
        tsv_path: tp.Union[str, Path] = "sets/musicraw-train.tsv",
        mean_std_path: tp.Union[str, Path, None] = None,
        sample_rate: int = 44_100,
        duration_sec: float = 5.0,
        audio_samples: tp.Optional[int] = None,
        normalize_audio: bool = False,
        stitch_transform_p: float = 0.25,
    ):
        super().__init__()

        self.root = Path(root)
        self.tsv_path = Path(tsv_path)
        if mean_std_path is None:
            mean_std_path = self.root / "mean_std" / f"{split}"
        self.mean_std_path = Path(mean_std_path)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        if audio_samples is None:
            audio_samples = int(sample_rate * duration_sec)
            # round to the next multiple of 1024
            audio_samples = ceil(audio_samples / 1024) * 1024
        else:
            audio_samples = int(audio_samples)
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 16ms
            assert (
                abs(effective_duration - duration_sec) < 0.016
            ), f"audio_samples {audio_samples} does not match duration_sec {duration_sec}"
        self.audio_samples = audio_samples
        self.normalize_audio = normalize_audio

        self.videos: list[str] = []
        self.labels: dict[str, str] = {}
        not_in_split: set[str] = set()
        videos = self._read_videos_from_root()
        meta = self._read_metadata()
        for video in videos:
            id = video
            label = meta.get(id)
            if label is None:
                log.debug(f"Video not in split: {video}")
                not_in_split.add(video)
                continue
            self.videos.append(video)
            self.labels[video] = label

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {root}")
            log.info(f"{len(self.videos)} videos found in {tsv_path}")
            log.debug(f"{len(not_in_split)} videos not in split but are in root")

        self.expected_audio_length = audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

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
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.mask_resize,
            ]
        )
        self.resampler: dict[int, torchaudio.transforms.Resample] = {}
        if split.lower() == "train":
            self.stitch_transform_p = stitch_transform_p
            self.video_transforms = v2.Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    ColorJitter(s=0.5),
                ]
            )
        else:
            self.stitch_transform_p = 0.0
            self.video_transforms = v2.Compose([IdentityTransform()])
        self.local_it_count = 0

    def _read_videos_from_root(self) -> set[str]:
        videos = set()
        for path in self.root.rglob("*.mp4"):
            videos.add(f"{path.parent.name}/{path.stem}")
        return videos

    def _read_metadata(self) -> dict:
        df = pd.read_csv(self.tsv_path, sep="\t", dtype={"id": str})
        df_dict = df.set_index("id")["label"].to_dict()
        return df_dict

    def _get_data_chunks(self, video: str) -> tp.Dict[str, tp.Any]:
        reader = StreamingMediaDecoder(self.root / (video + ".mp4"))
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
            frames_per_chunk=2**30,
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = self.clip_transform(data_chunk[0])[: self.clip_expected_length]
        sync_chunk = self.sync_transform(data_chunk[1])[: self.sync_expected_length]
        audio_chunk = data_chunk[2]
        data_chunks = {
            "clip": clip_chunk,
            "sync": sync_chunk,
            "audio": audio_chunk,
            "audio_fps": reader.get_out_stream_info(2).sample_rate,
        }
        return data_chunks

    def _process_audio(
        self, audio_chunk: torch.Tensor, sample_rate: int, video: str
    ) -> torch.Tensor:
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)  # mono
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
            if abs_max <= 1e-6:
                raise RuntimeError(f"Audio is silent {video}")
        if sample_rate == self.sample_rate:
            audio_chunk = audio_chunk
        else:
            if sample_rate not in self.resampler:
                # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
            audio_chunk = self.resampler[sample_rate](audio_chunk)
        return audio_chunk

    def _interpolate_mask(self, mask_chunk: torch.Tensor) -> torch.Tensor:
        mask_chunk = mask_chunk.permute(1, 2, 0)
        mask_chunk = interpolate(
            mask_chunk, size=self.sync_expected_length, mode="nearest-exact"
        )
        mask_chunk = mask_chunk.permute(2, 0, 1)
        mask_chunk = mask_chunk.to(torch.bool)
        return mask_chunk

    def _load_png_to_tensor(
        self, image_path: str, convert_type: str = "L", expand_mask: bool = False
    ) -> torch.Tensor:
        image = Image.open(image_path).convert(convert_type)
        image_t = self.mask_transform(image)
        return image_t

    def _get_mask_sequence(self, video: str) -> tp.Optional[torch.Tensor]:
        mask_path = self.root / video / "propagated"
        if not mask_path.exists():
            return None
        ret = []
        for mask_fn in sorted(mask_path.glob("*.png")):
            ret.append(self._load_png_to_tensor(mask_fn.as_posix(), convert_type="L"))
        mask_chunk = torch.stack(ret, dim=0)
        if mask_chunk.shape[0] != self.sync_expected_length:
            mask_chunk = self._interpolate_mask(mask_chunk)
        return mask_chunk

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        td = TensorDict.load_memmap(self.mean_std_path)
        mean = td["mean"]
        return mean.mean(dim=(0, 1)), mean.std(dim=(0, 1))

    def __len__(self) -> int:
        return len(self.videos)

    def _stitch_transform(
        self,
        data_chunks: tp.Dict[str, tp.Any],
        mask_chunk: torch.Tensor,
        idx: int,
    ) -> tp.Tuple[tp.Dict[str, tp.Any], torch.Tensor, str, str]:
        loaded_video = False
        max_try = 10
        while not loaded_video and max_try > 0:
            max_try -= 1
            # randomly select a video to stitch
            stitch_idx = torch.randint(0, len(self.videos), (1,)).item()
            if stitch_idx == idx:
                continue
            stitch_video = self.videos[stitch_idx]  # type: ignore
            try:
                stitch_data_chunks = self._get_data_chunks(stitch_video)
                stitch_mask_chunk = self._get_mask_sequence(stitch_video)
                if stitch_mask_chunk is None:
                    continue
                self._check_chunks(
                    stitch_data_chunks,
                    stitch_mask_chunk,
                )
                loaded_video = True
            except Exception as e:
                continue

        if not loaded_video:
            raise RuntimeError(f"Failed to load video {stitch_video}")
        # stitch the two videos spatially side-by-side
        dim = choice([2, 3])
        clip_chunk = torch.cat(
            [data_chunks["clip"], stitch_data_chunks["clip"]], dim=dim
        )
        sync_chunk = torch.cat(
            [data_chunks["sync"], stitch_data_chunks["sync"]], dim=dim
        )
        # randomly select audio and mask from same video
        if torch.rand(1) < 0.5:
            video = stitch_video
            label = self.labels[stitch_video]
            audio_chunk = stitch_data_chunks["audio"]
            audio_fps = stitch_data_chunks["audio_fps"]
            empty_mask = torch.zeros_like(mask_chunk)
            mask_chunk = torch.cat([empty_mask, stitch_mask_chunk], dim=dim)  # type: ignore
        else:
            video = self.videos[idx]
            label = self.labels[video]
            audio_chunk = data_chunks["audio"]
            audio_fps = data_chunks["audio_fps"]
            empty_mask = torch.zeros_like(stitch_mask_chunk)  # type: ignore
            mask_chunk = torch.cat([mask_chunk, empty_mask], dim=dim)
        return (
            {
                "clip": self.clip_resize(clip_chunk),
                "sync": self.sync_resize(sync_chunk),
                "audio": audio_chunk,
                "audio_fps": audio_fps,
            },
            self.mask_resize(mask_chunk),
            video,
            label,
        )

    def _check_chunks(
        self,
        data_chunks: tp.Dict[str, torch.Tensor],
        mask_chunk: torch.Tensor,
    ):
        clip_chunk = data_chunks["clip"]
        sync_chunk = data_chunks["sync"]
        audio_chunk = data_chunks["audio"]
        if clip_chunk is None:
            raise RuntimeError(f"CLIP video returned None")
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video too short, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}"
            )
        if sync_chunk is None:
            raise RuntimeError(f"Sync video returned None")
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f"Sync video too short, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}"
            )
        if mask_chunk is None:
            raise RuntimeError(f"Mask video not found")
        if audio_chunk is None:
            raise RuntimeError(f"Audio returned None")
        if audio_chunk.shape[0] == 0:
            raise RuntimeError(f"Audio is empty")

        return True

    def sample(self, idx: int) -> dict[str, tp.Any]:
        self.local_it_count += 1
        video = self.videos[idx]
        label = self.labels[video]
        # read
        data_chunks = self._get_data_chunks(video)
        mask_chunk = self._get_mask_sequence(video)
        if mask_chunk is None:
            raise RuntimeError(f"Mask video not found for {video}")
        self._check_chunks(data_chunks, mask_chunk)

        if (
            torch.rand(1) < self.stitch_transform_p
            and self.local_it_count > STITCH_TRANSFORM_WARMUP
        ):
            data_chunks, mask_chunk, video, label = self._stitch_transform(
                data_chunks, mask_chunk, idx
            )

        clip_chunk = data_chunks["clip"]
        sync_chunk = data_chunks["sync"]
        audio_chunk = data_chunks["audio"]
        audio_fps = data_chunks["audio_fps"]

        # process audio
        audio_chunk = self._process_audio(audio_chunk, audio_fps, video)
        if audio_chunk.shape[0] < self.expected_audio_length:
            raise RuntimeError(
                f"Audio too short {video}, expected {self.expected_audio_length}, got {audio_chunk.shape[0]}"
            )
        audio_chunk = audio_chunk[: self.expected_audio_length]

        datapoint = {
            "id": video,
            "caption": label.replace("playing_", ""),
            "audio": audio_chunk,
            "clip_video": clip_chunk,
            "sync_video": sync_chunk,
            "mask_video": mask_chunk,
            "name": Path(video).name,
        }
        return self.video_transforms(datapoint)

    def __getitem__(self, idx: int) -> tp.Optional[dict[str, torch.Tensor]]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None
