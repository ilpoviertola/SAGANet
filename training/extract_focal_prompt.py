import logging
import typing as tp
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

from tqdm import tqdm
import einops
import numpy as np
import av
from PIL import Image
import torch
import torch.distributed as distributed
import torch.distributed.elastic
import torch.distributed.elastic.multiprocessing
import torch.distributed.elastic.multiprocessing.errors
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.ops import masks_to_boxes
from torchvision.transforms.v2 import Resize, Compose, Identity

from mmaudio.data.data_setup import error_avoidance_collate
from mmaudio.data.raw_music import MusicRaw as Music
from mmaudio.data.extraction.urmp import URMPDataset as URMP
from mmaudio.data.extraction.segmented_music_solos import SegmentedMusicSolos
from mmaudio.utils.dist_utils import local_rank, world_size

"""
NOTE: 220500 (5*44100) is not divisible by (STFT hop size * VAE downsampling ratio) which is 1024.
221184 is the next integer divisible by 1024.
"""

DATASET = "segmented_music_solos"
SAMPLING_RATE = 44100
DURATION_SEC = 5.0
NUM_SAMPLES = 221184

# per-GPU
BATCH_SIZE = 3
NUM_WORKERS = 8

MIN_CROP_W = 56
MIN_CROP_H = 56

log = logging.getLogger()
log.setLevel(logging.INFO)

# uncomment the train/test/val sets to extract latents for them
## urmp
# data_cfg = {
#     "test": {
#         "root": "./data/urmp",
#         "tsv": "./data/urmp/metadata.tsv",
#         "normalize_audio": False,
#     }
# }
## segmened music solos (sms)
data_cfg = {
    "train": {
        "root": "./data/sms",
        "tsv": "./data/sms/metadata.csv",
        "split": "train",
        "normalize_audio": False,
    },
    "val": {
        "root": "./data/sms",
        "tsv": "./data/sms/metadata.csv",
        "split": "val",
        "normalize_audio": False,
    },
}


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size


def setup_dataset(split: str):
    if DATASET == "music":
        dataset = Music(  # type: ignore
            root=str(data_cfg[split]["root"]),
            tsv_path=str(data_cfg[split]["tsv"]),
            split="test",  # hack
            sample_rate=SAMPLING_RATE,
            duration_sec=DURATION_SEC,
            audio_samples=NUM_SAMPLES,
            normalize_audio=bool(data_cfg[split]["normalize_audio"]),
        )
    elif DATASET == "urmp":
        dataset = URMP(  # type: ignore
            root=str(data_cfg[split]["root"]),
            tsv_path=str(data_cfg[split]["tsv"]),
            sample_rate=SAMPLING_RATE,
            duration_sec=DURATION_SEC,
            audio_samples=NUM_SAMPLES,
            normalize_audio=bool(data_cfg[split]["normalize_audio"]),
        )
        dataset.mask_transform = Compose([Identity()])
        dataset.sync_transform = Compose([Identity()])
    elif DATASET == "segmented_music_solos":
        dataset = SegmentedMusicSolos(  # type: ignore
            root=str(data_cfg[split]["root"]),
            tsv_path=str(data_cfg[split]["tsv"]),
            sample_rate=SAMPLING_RATE,
            duration_sec=DURATION_SEC,
            audio_samples=NUM_SAMPLES,
            normalize_audio=bool(data_cfg[split]["normalize_audio"]),
        )
        dataset.mask_transform = Compose([dataset.mask_resize])
        dataset.sync_transform = Compose([dataset.sync_resize])
    else:
        raise NotImplementedError(
            f"Dataset {DATASET} is not implemented. Please check the code."
        )

    sampler: DistributedSampler = DistributedSampler(
        dataset, rank=local_rank, shuffle=False
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=sampler,
        drop_last=False,
        collate_fn=error_avoidance_collate,
    )
    return dataset, loader


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: tp.Optional[tp.Dict[str, tp.Any]] = None,
    audio_array: tp.Optional[torch.Tensor] = None,
    audio_fps: tp.Optional[float] = None,
    audio_codec: tp.Optional[str] = None,
    audio_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()  # type: ignore

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)  # type: ignore
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        # stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.pix_fmt = "yuv420p" if video_array.shape[3] == 3 else "gray"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)  # type: ignore
            a_stream.options = audio_options or {}  # type: ignore

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = a_stream.format.name  # type: ignore

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)  # type: ignore

            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout  # type: ignore
            )

            frame.sample_rate = audio_fps  # type: ignore

            for packet in a_stream.encode(frame):  # type: ignore
                container.mux(packet)

            for packet in a_stream.encode():  # type: ignore
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format=("gray" if video_array.shape[3] == 1 else "rgb24"))  # type: ignore
            frame.pict_type = 0  # type: ignore
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def save_video(
    audio: torch.Tensor,
    frames: torch.Tensor,
    output_dir_path: Path,
    fn: str,
    v_fps: float = 25,
    a_fps: int = 44100,
):
    if fn.endswith(".mp4") or fn.endswith(".wav"):
        fn = fn[:-4]
    video_path = output_dir_path / f"{fn}.mp4"

    if video_path.exists():
        print("File %s already exists. Overwriting...", video_path.as_posix())
    write_video(
        filename=video_path.as_posix(),
        video_array=frames.cpu().numpy(),  # type: ignore
        fps=v_fps,
        video_codec="h264",
        options={"crf": "10", "pix_fmt": "yuv420p"},
        audio_array=audio,
        audio_fps=a_fps,
        audio_codec="aac",
    )


@torch.distributed.elastic.multiprocessing.errors.record
@torch.inference_mode()
def run_focal_crop(
    x: torch.Tensor, x_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # x: (B, T, C, H, W) H/W: 224

    B, T, c, H, W = x.shape
    assert c == 3 and H == 224 and W == 224
    Bm, Tm, cm, Hm, Wm = x_mask.shape
    assert cm == 1 and Hm == 224 and Wm == 224
    assert B == Bm and T == Tm

    resize = Resize((H, W), antialias=True)
    detail_x = torch.zeros_like(x, dtype=x.dtype)
    detail_mask = torch.zeros_like(x_mask, dtype=x_mask.dtype)
    for b in range(B):
        cur_mask = einops.rearrange(x_mask[b], "T C H W -> (T C) H W")
        n = cur_mask.shape[0]
        bbox = torch.zeros((n, 4), device=cur_mask.device, dtype=torch.float)
        for index, mask in enumerate(cur_mask):
            if mask.numel() == 0:
                bbox[index] = torch.tensor([0, 0, 0, 0], device=mask.device)
                continue
            elif mask.sum() == 0:
                bbox[index] = torch.tensor([0, 0, 0, 0], device=mask.device)
                continue
            bbox[index] = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)

        w, h = (
            torch.clamp(bbox[..., 2] - bbox[..., 0], min=MIN_CROP_W).max(),
            torch.clamp(bbox[..., 3] - bbox[..., 1], min=MIN_CROP_H).max(),
        )

        x0, y0 = (
            torch.round(bbox[..., 0]).int(),
            torch.round(bbox[..., 1]).int(),
        )

        x1, y1 = (
            torch.round(x0 + w).int(),
            torch.round(y0 + h).int(),
        )
        x1_res = torch.where(x1 > W, x1 - W, torch.zeros_like(x1))
        x1 = (x1 - x1_res).max()
        y1_res = torch.where(y1 > H, y1 - H, torch.zeros_like(y1))
        y1 = (y1 - y1_res).max()

        x0 = (x0 - x1_res).min()
        y0 = (y0 - y1_res).min()

        # assert x0.dim() == x1.dim() == y0.dim() == y1.dim() == 1
        assert x0.min() >= 0 and x1.max() <= W
        assert y0.min() >= 0 and y1.max() <= H
        i = 0
        for t in range(T):
            # im = x[b, t, :, y0[i] : y1[i], x0[i] : x1[i]]
            im = x[b, t, :, y0:y1, x0:x1]
            im = resize(im)
            detail_x[b, t, :, :, :] = im
            # m = x_mask[b, t, :, y0[i] : y1[i], x0[i] : x1[i]]
            m = x_mask[b, t, :, y0:y1, x0:x1]
            m = resize(m)
            detail_mask[b, t, :, :, :] = m
            i += 1

    return detail_x, detail_mask


def get_urmp_fn(fn: str, mask_fn: str) -> str:
    folder, fn = fn.split("/")
    fn_parts = fn.split("_")
    song_num = fn_parts[1]
    song_name = fn_parts[2]
    s_ts = fn_parts[-2]
    e_ts = fn_parts[-1]
    maskfn_parts = mask_fn.split("/")[1].split("_")
    mask_num = maskfn_parts[1]
    mask_name = maskfn_parts[2]
    fn = f"{folder}/Vid_{song_num}_{song_name}_{mask_num}_{mask_name}_{s_ts}_{e_ts}"
    return fn


@torch.distributed.elastic.multiprocessing.errors.record
@torch.inference_mode()
def extract():
    # initial setup
    distributed_setup()

    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=Path, default="./training/example_output/focal_prompt"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in data_cfg.keys():
        # setup datasets
        dataset, loader = setup_dataset(split)
        log.info(f"Number of samples: {len(dataset)}")
        log.info(f"Number of batches: {len(loader)}")

        for _, data in enumerate(tqdm(loader)):
            try:
                sync_video = data["sync_video"]
                mask_video = data["mask_video"]
                cropped_x, cropped_mask = run_focal_crop(sync_video, mask_video)
            except Exception as e:
                print(f"Error in batch ({data['id']}): {e}")
                continue

            if "mask_id" not in data:
                data["mask_id"] = data["id"]
            # save the cropped videos
            for video, mask, fn, mask_fn in zip(
                cropped_x, cropped_mask, data["id"], data["mask_id"]
            ):
                try:
                    if DATASET == "urmp":
                        fn = get_urmp_fn(fn, mask_fn)
                    filepath = output_dir / fn / f"{Path(fn).name}_detail.mp4"
                    if not filepath.exists():
                        video = video.permute(0, 2, 3, 1)
                        write_video(
                            filename=filepath.as_posix(),
                            video_array=video,
                            fps=25,
                        )
                    # save the mask
                    if DATASET in ["urmp", "segmented_music_solos"]:
                        mask_filepath = (
                            output_dir / fn / f"{Path(fn).name}_mask_detail.mp4"
                        )
                        if not mask_filepath.exists():
                            mask = mask.permute(0, 2, 3, 1)
                            write_video(
                                filename=mask_filepath.as_posix(),
                                video_array=mask,
                                fps=25,
                            )
                    else:
                        mask_filepath = output_dir / fn / "propagated_detail"
                        mask = einops.rearrange(mask, "T C H W -> (T C) H W")
                        for i, m in enumerate(mask):
                            mask_img = Image.fromarray(m.numpy(), mode="L")
                            mask_img.save(mask_filepath / f"{i:04d}.png")
                except Exception as e:
                    print(f"Error saving video ({fn}): {e}")
                    continue


if __name__ == "__main__":
    extract()
    distributed.destroy_process_group()
