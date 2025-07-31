import json
import logging
import os
from pathlib import Path
from typing import Union
from math import ceil

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from PIL import Image

from saganet.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


class VideoDataset(Dataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_root = Path(video_root)

        self.duration_sec = duration_sec

        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose(
            [
                v2.Resize(
                    (_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.sync_transform = v2.Compose(
            [
                v2.Resize(
                    (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
                # v2.CenterCrop(_SYNC_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # to be implemented by subclasses
        self.captions = {}
        self.videos = sorted(list(self.captions.keys()))

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        caption = self.captions[video_id]

        reader = StreamingMediaDecoder(self.video_root / (video_id + ".mp4"))
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

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        if clip_chunk is None:
            raise RuntimeError(f"CLIP video returned None {video_id}")
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}"
            )

        if sync_chunk is None:
            raise RuntimeError(f"Sync video returned None {video_id}")
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f"Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}"
            )

        # truncate the video
        clip_chunk = clip_chunk[: self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video wrong length {video_id}, "
                f"expected {self.clip_expected_length}, "
                f"got {clip_chunk.shape[0]}"
            )
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[: self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(
                f"Sync video wrong length {video_id}, "
                f"expected {self.sync_expected_length}, "
                f"got {sync_chunk.shape[0]}"
            )
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            "name": video_id,
            "caption": caption,
            "clip_video": clip_chunk,
            "sync_video": sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def __len__(self):
        return len(self.captions)


class VGGSound(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        *,
        duration_sec: float = 8.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.csv_path = Path(csv_path)

        videos = sorted(os.listdir(self.video_root))
        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
        self.captions = {}

        df = pd.read_csv(
            csv_path, header=None, names=["id", "sec", "caption", "split"]
        ).to_dict(orient="records")

        videos_no_found = []
        for row in df:
            if row["split"] == "test":
                start_sec = int(row["sec"])
                video_id = str(row["id"])
                # this is how our videos are named
                video_name = f"{video_id}_{start_sec:06d}"
                if video_name + ".mp4" not in videos:
                    videos_no_found.append(video_name)
                    continue

                self.captions[video_name] = row["caption"]

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
            log.info(f"{len(self.captions)} useable videos found")
            if videos_no_found:
                log.info(
                    f"{len(videos_no_found)} found in {csv_path} but not in {video_root}"
                )
                log.info(
                    "A small amount is expected, as not all videos are still available on YouTube"
                )

        self.videos = sorted(list(self.captions.keys()))


class MovieGen(VideoDataset):

    def __init__(
        self,
        video_root: Union[str, Path],
        jsonl_root: Union[str, Path],
        *,
        duration_sec: float = 10.0,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.video_root = Path(video_root)
        self.jsonl_root = Path(jsonl_root)

        videos = sorted(os.listdir(self.video_root))
        videos = [v[:-4] for v in videos]  # remove extensions
        self.captions = {}

        for v in videos:
            with open(self.jsonl_root / (v + ".jsonl")) as f:
                data = json.load(f)
                self.captions[v] = data["audio_prompt"]

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")

        self.videos = videos


class AVSSemantic(VideoDataset):
    _EXPANSION_PIX = 5

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        mask_root: Union[str, Path],
        *,
        duration_sec: float = 5,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.csv_path = Path(csv_path)
        self.mask_root = Path(mask_root)
        videos = sorted(list(Path(video_root).rglob("*.mp4")))
        videos = [f"{v.parent.name}/{v.name}" for v in videos]  # type: ignore
        if local_rank == 0:
            log.info(f"{len(self.videos)} videos found in {video_root}")

        df = pd.read_csv(
            csv_path, header=None, names=["name", "start", "category", "split"]
        ).to_dict(orient="records")

        videos_no_found = []
        for row in df:
            if row["split"] == "test":
                video_name = row["name"]
                if row["category"] + "/" + video_name + ".mp4" not in videos:
                    videos_no_found.append(video_name)
                    continue

                self.captions[f"{row['category']}/{video_name}"] = row["category"]

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
            log.info(f"{len(self.captions)} useable videos found")
            if videos_no_found:
                log.info(
                    f"{len(videos_no_found)} found in {csv_path} but not in {video_root}"
                )
                log.info(
                    "A small amount is expected, as not all videos are still available on YouTube"
                )
        self.videos = sorted(list(self.captions.keys()))

        self.mask_video_transform = v2.Compose(
            [
                v2.UniformTemporalSubsample(self.sync_expected_length),
                v2.Resize(
                    (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
                # v2.CenterCrop(_SYNC_SIZE),
            ]
        )
        self.to_tensor_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if data is None:  # error loading video
            return None
        try:
            video_id = data["name"]
            mask_chunk = self.get_extended_masks(self.mask_root / video_id)
            mask_chunk = self.mask_video_transform(mask_chunk)
            mask_chunk = mask_chunk[: self.sync_expected_length]
            if mask_chunk.shape[0] != self.sync_expected_length:
                raise RuntimeError(
                    f"Mask video wrong length {video_id}, "
                    f"expected {self.sync_expected_length}, "
                    f"got {mask_chunk.shape[0]}"
                )
            data["mask_video"] = mask_chunk
            return data
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def get_extended_masks(self, sample_dir: Path) -> torch.Tensor:
        """Get extended masks for the sample.

        Args:
            item (dict): Single sample.

        Returns:
            torch.Tensor: Extended masks.
        """
        mask_dir = sample_dir / "extended_mask"
        if not mask_dir.exists() and not mask_dir.is_file():
            raise FileNotFoundError(f"Extended mask dir not found: {mask_dir}")

        ret = []
        for mask_file in sorted(
            mask_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            ret.append(self._load_png_to_tensor(mask_file.as_posix(), convert_type="L"))
        mask_video = torch.stack(ret, dim=0)  # (T, C, H, W)
        return mask_video

    def _load_png_to_tensor(
        self, image_path: str, convert_type: str = "L", expand_mask: bool = False
    ) -> torch.Tensor:
        image = Image.open(image_path).convert(convert_type)
        image_t = self.to_tensor_transform(image)
        if expand_mask:
            # Convert to binary mask
            binary_mask = image_t > 0.5
            if not torch.any(binary_mask):
                return image_t
            if binary_mask.ndim == 3:
                binary_mask = binary_mask.squeeze(0)
            # Find bounding box
            non_zero_indices = torch.nonzero(binary_mask)
            min_y, min_x = torch.min(non_zero_indices, dim=0)[0]
            max_y, max_x = torch.max(non_zero_indices, dim=0)[0]
            # Expand bounding box by _EXPANSION_PIX pixels
            min_y = max(min_y - self._EXPANSION_PIX, 0)
            min_x = max(min_x - self._EXPANSION_PIX, 0)
            max_y = min(max_y + self._EXPANSION_PIX, image_t.shape[1] - 1)
            max_x = min(max_x + self._EXPANSION_PIX, image_t.shape[2] - 1)
            # Create new mask with expanded bounding box
            image_t = torch.zeros_like(image_t)
            image_t[:, min_y : max_y + 1, min_x : max_x + 1] = 1

        return image_t


class MultiSource(VideoDataset):
    _EXPANSION_PIX = 5

    def __init__(
        self,
        video_root: Union[str, Path],
        csv_path: Union[str, Path],
        mask_root: Union[str, Path],
        use_captions: bool = False,
        *,
        duration_sec: float = 5,
    ):
        super().__init__(video_root, duration_sec=duration_sec)
        self.csv_path = Path(csv_path)
        self.mask_root = Path(mask_root)
        self.use_captions = use_captions
        videos = sorted(list(Path(video_root).rglob("*.mp4")))
        videos = [f"{v.parent.name}/{v.name}" for v in videos]  # type: ignore
        if local_rank == 0:
            log.info(f"{len(self.videos)} videos found in {video_root}")

        df = pd.read_csv(
            csv_path,
            header=None,
            names=["video_id", "obj_id", "semantic_label", "audio_action_label"],
        ).to_dict(orient="records")

        videos_no_found = []
        videos_found = []
        for row in df:
            video_name = row["video_id"]
            if video_name + "/" + video_name + ".mp4" not in videos:
                videos_no_found.append(video_name)
                continue

            video_id = f"{video_name}/{video_name}"
            if video_id not in self.captions:
                self.captions[video_id] = [
                    f"{row['obj_id']}: {row['semantic_label']}, {row['audio_action_label']}"
                ]
            else:
                self.captions[video_id].append(
                    f"{row['obj_id']}: {row['semantic_label']}, {row['audio_action_label']}"
                )

            videos_found.append(
                {
                    "video_id": video_id,
                    "obj_id": row["obj_id"],
                    "semantic_label": row["semantic_label"],
                    "audio_action_label": row["audio_action_label"],
                    "caption": f"{row['semantic_label']}, {row['audio_action_label']}",
                }
            )

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {video_root}")
            log.info(f"{len(videos_found)} useable videos found with different objects")
            if videos_no_found:
                log.info(
                    f"{len(videos_no_found)} found in {csv_path} but not in {video_root}"
                )
                log.info(
                    "A small amount is expected, as not all videos are still available on YouTube"
                )
                log.info("Videos: " + str(videos_no_found))
        self.videos = videos_found

        self.mask_video_transform = v2.Compose(
            [
                v2.UniformTemporalSubsample(self.sync_expected_length),
                v2.Resize(
                    (_SYNC_SIZE, _SYNC_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
            ]
        )
        self.to_tensor_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            video_id = self.videos[idx]["video_id"]

            reader = StreamingMediaDecoder(self.video_root / (video_id + ".mp4"))
            video_fps = reader.get_src_stream_info(0).frame_rate

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

            clip_chunk = data_chunk[0]
            sync_chunk = data_chunk[1]
            if clip_chunk is None:
                raise RuntimeError(f"CLIP video returned None {video_id}")
            if clip_chunk.shape[0] < self.clip_expected_length:
                raise RuntimeError(
                    f"CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}"
                )

            if sync_chunk is None:
                raise RuntimeError(f"Sync video returned None {video_id}")
            if sync_chunk.shape[0] < self.sync_expected_length:
                raise RuntimeError(
                    f"Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}"
                )

            # truncate the video
            clip_chunk = clip_chunk[: self.clip_expected_length]
            if clip_chunk.shape[0] != self.clip_expected_length:
                raise RuntimeError(
                    f"CLIP video wrong length {video_id}, "
                    f"expected {self.clip_expected_length}, "
                    f"got {clip_chunk.shape[0]}"
                )
            clip_chunk = self.clip_transform(clip_chunk)

            sync_chunk = sync_chunk[: self.sync_expected_length]
            if sync_chunk.shape[0] != self.sync_expected_length:
                raise RuntimeError(
                    f"Sync video wrong length {video_id}, "
                    f"expected {self.sync_expected_length}, "
                    f"got {sync_chunk.shape[0]}"
                )
            sync_chunk = self.sync_transform(sync_chunk)

            data = {}
            data["clip_video"] = clip_chunk
            data["sync_video"] = sync_chunk
            if self.use_captions:
                data["caption"] = self.videos[idx]["caption"]

            obj_id = f"obj{self.videos[idx]['obj_id']}"
            video_id = video_id.split("/")[0]
            sem_label = self.videos[idx]["semantic_label"]
            data["name"] = f"{video_id}/{video_id}_{obj_id}-{sem_label}"

            mask_chunk = self.get_extended_masks(
                self.mask_root / video_id,
                obj_id,
                limit=ceil(video_fps * self.duration_sec),
            )
            mask_chunk = self.mask_video_transform(mask_chunk)
            if mask_chunk.shape[0] != self.sync_expected_length:
                raise RuntimeError(
                    f"Mask video wrong length {video_id}, "
                    f"expected {self.sync_expected_length}, "
                    f"got {mask_chunk.shape[0]}"
                )
            data["mask_video"] = mask_chunk
            return data
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def get_extended_masks(
        self, sample_dir: Path, obj_id: str, limit=None
    ) -> torch.Tensor:
        """Get extended masks for the sample.

        Args:
            item (dict): Single sample.

        Returns:
            torch.Tensor: Extended masks.
        """
        mask_dir = sample_dir / obj_id
        if not mask_dir.exists() and not mask_dir.is_file():
            raise FileNotFoundError(f"Extended mask dir not found: {mask_dir}")

        ret = []
        i = 0
        for mask_file in sorted(mask_dir.glob("*.png"), key=lambda x: int(x.stem)):
            ret.append(self._load_png_to_tensor(mask_file.as_posix(), convert_type="L"))
            i += 1
            if limit and i >= limit:
                break
        mask_video = torch.stack(ret, dim=0)  # (T, C, H, W)
        return mask_video

    def _load_png_to_tensor(
        self, image_path: str, convert_type: str = "L", expand_mask: bool = False
    ) -> torch.Tensor:
        image = Image.open(image_path).convert(convert_type)
        image_t = self.to_tensor_transform(image)
        if expand_mask:
            # Convert to binary mask
            binary_mask = image_t > 0.5
            if not torch.any(binary_mask):
                return image_t
            if binary_mask.ndim == 3:
                binary_mask = binary_mask.squeeze(0)
            # Find bounding box
            non_zero_indices = torch.nonzero(binary_mask)
            min_y, min_x = torch.min(non_zero_indices, dim=0)[0]
            max_y, max_x = torch.max(non_zero_indices, dim=0)[0]
            # Expand bounding box by _EXPANSION_PIX pixels
            min_y = max(min_y - self._EXPANSION_PIX, 0)
            min_x = max(min_x - self._EXPANSION_PIX, 0)
            max_y = min(max_y + self._EXPANSION_PIX, image_t.shape[1] - 1)
            max_x = min(max_x + self._EXPANSION_PIX, image_t.shape[2] - 1)
            # Create new mask with expanded bounding box
            image_t = torch.zeros_like(image_t)
            image_t[:, min_y : max_y + 1, min_x : max_x + 1] = 1

        return image_t
