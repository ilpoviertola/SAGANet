from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
from math import ceil, floor

import av
import numpy as np
import torch
from av import AudioFrame
from torchvision.io import _video_opt
from torchvision.io.video import (
    _check_av_available,
    _read_from_stream,
    _align_audio_frames,
)


@dataclass
class VideoInfo:
    duration_sec: float
    fps: Fraction
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    all_frames: Optional[list[np.ndarray]]

    @property
    def height(self):
        return self.all_frames[0].shape[0]

    @property
    def width(self):
        return self.all_frames[0].shape[1]

    @classmethod
    def from_image_info(
        cls, image_info: "ImageInfo", duration_sec: float, fps: Fraction
    ) -> "VideoInfo":
        num_frames = int(duration_sec * fps)
        all_frames = [image_info.original_frame] * num_frames
        return cls(
            duration_sec=duration_sec,
            fps=fps,
            clip_frames=image_info.clip_frames,
            sync_frames=image_info.sync_frames,
            all_frames=all_frames,
        )


@dataclass
class ImageInfo:
    clip_frames: torch.Tensor
    sync_frames: torch.Tensor
    original_frame: Optional[np.ndarray]

    @property
    def height(self):
        return self.original_frame.shape[0]

    @property
    def width(self):
        return self.original_frame.shape[1]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def read_frames(
    video_path: Path,
    list_of_fps: list[float],
    start_sec: float,
    end_sec: float,
    need_all_frames: bool,
    v_format: str,
) -> tuple[list[np.ndarray], list[np.ndarray], Fraction]:
    output_frames = [[] for _ in list_of_fps]
    next_frame_time_for_each_fps = [0.0 for _ in list_of_fps]
    time_delta_for_each_fps = [1 / fps for fps in list_of_fps]
    all_frames = []

    # container = av.open(video_path)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = stream.guessed_rate
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time < start_sec:
                    continue
                if frame_time > end_sec:
                    break

                frame_np = None
                if need_all_frames:
                    frame_np = frame.to_ndarray(format="rgb24")
                    all_frames.append(frame_np)

                for i, _ in enumerate(list_of_fps):
                    this_time = frame_time
                    while this_time >= next_frame_time_for_each_fps[i]:
                        if frame_np is None:
                            frame_np = frame.to_ndarray(format="rgb24")

                        output_frames[i].append(frame_np)
                        next_frame_time_for_each_fps[i] += time_delta_for_each_fps[i]

    ret = []
    for frames in output_frames:
        frames = np.stack(frames)
        if v_format == "gray":
            frames = rgb2gray(frames)
            frames = frames[:, np.newaxis, ...]
            # frames to uint8
            frames = (frames * 255).astype(np.uint8)
        ret.append(frames)
    return ret, all_frames, fps


def reencode_with_audio(
    video_info: VideoInfo, output_path: Path, audio: torch.Tensor, sampling_rate: int
):
    container = av.open(output_path, "w")
    output_video_stream = container.add_stream("h264", video_info.fps)
    output_video_stream.codec_context.bit_rate = 10 * 1e6  # 10 Mbps
    output_video_stream.width = video_info.width
    output_video_stream.height = video_info.height
    output_video_stream.pix_fmt = "yuv420p"

    output_audio_stream = container.add_stream("aac", sampling_rate)

    # encode video
    for image in video_info.all_frames:
        image = av.VideoFrame.from_ndarray(image)
        packet = output_video_stream.encode(image)
        container.mux(packet)

    for packet in output_video_stream.encode():
        container.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    audio_frame = AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        container.mux(packet)

    for packet in output_audio_stream.encode():
        container.mux(packet)

    container.close()


def remux_with_audio(
    video_path: Path, audio: torch.Tensor, output_path: Path, sampling_rate: int
):
    """
    NOTE: I don't think we can get the exact video duration right without re-encoding
    so we are not using this but keeping it here for reference
    """
    video = av.open(video_path)
    output = av.open(output_path, "w")
    input_video_stream = video.streams.video[0]
    output_video_stream = output.add_stream(template=input_video_stream)
    output_audio_stream = output.add_stream("aac", sampling_rate)

    duration_sec = audio.shape[-1] / sampling_rate

    for packet in video.demux(input_video_stream):
        # We need to skip the "flushing" packets that `demux` generates.
        if packet.dts is None:
            continue
        # We need to assign the packet to the new stream.
        packet.stream = output_video_stream
        output.mux(packet)

    # convert float tensor audio to numpy array
    audio_np = audio.numpy().astype(np.float32)
    audio_frame = av.AudioFrame.from_ndarray(audio_np, format="flt", layout="mono")
    audio_frame.sample_rate = sampling_rate

    for packet in output_audio_stream.encode(audio_frame):
        output.mux(packet)

    for packet in output_audio_stream.encode():
        output.mux(packet)

    video.close()
    output.close()

    output.close()


def read_video_to_frames_and_audio_with_av(
    path: Path,
    v_start_s: Union[float, Fraction] = 0,
    v_end_s: Optional[Union[float, Fraction]] = None,
    a_start_s: Optional[Union[float, Fraction]] = None,
    a_end_s: Optional[Union[float, Fraction]] = None,
    to_rgb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    with av.open(path.as_posix(), metadata_errors="ignore") as av_container:
        if av_container.streams.video:
            video = av_container.streams.video[0]
            v_duration = float(video.duration * video.time_base)
            vfps = float(video.average_rate)
        else:
            v_duration = float("inf")
            vfps = None

        if av_container.streams.audio:
            audio = av_container.streams.audio[0]
            a_end_s = (
                a_end_s + 2 / vfps if a_end_s is not None and vfps is not None else None
            )
            a_duration = float(audio.duration * audio.time_base)
        else:
            a_duration = float("inf")

        duration_s = min(v_duration, a_duration)
        rgb, audio, meta = parse_av_container(
            av_container,
            v_start_s,
            v_end_s,
            a_start_s,
            a_end_s,
            pts_unit="sec",
            output_format="TCHW",
            to_rgb=to_rgb,
        )
        meta["duration"] = duration_s
    return rgb, audio, meta


def parse_av_container(
    container,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    audio_start_pts: Optional[Union[float, Fraction]] = None,
    audio_end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "TCHW",
    to_rgb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames.
    Extended from https://pytorch.org/vision/main/generated/torchvision.io.read_video.html
        to parameterise the audio start and end pts.

    Args:
        container (TODO type): opened container
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        audio_start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the audio
        audio_end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time of the audio
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors.
                                       Can be either "TCHW" (default) or "THWC".
        to_rgb (bool, optional): Whether to convert video frames to RGB. Defaults to True.

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if audio_start_pts is None:
        audio_start_pts = start_pts
    if audio_end_pts is None:
        audio_end_pts = end_pts

    if end_pts < start_pts:
        raise ValueError(
            f"end_pts should be > than start_pts, got start={start_pts} and end={end_pts}"
        )

    info = {}
    video_frames = []
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    if container.streams.audio:
        audio_timebase = container.streams.audio[0].time_base
    if container.streams.video:
        video_frames = _read_from_stream(
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
        )
        video_fps = container.streams.video[0].average_rate
        # guard against potentially corrupted files
        if video_fps is not None:
            info["video_fps"] = float(video_fps)

    if container.streams.audio:
        audio_frames = _read_from_stream(
            container,
            audio_start_pts,
            audio_end_pts,
            pts_unit,
            container.streams.audio[0],
            {"audio": 0},
        )
        info["audio_fps"] = container.streams.audio[0].rate

    vframes_list = [
        frame.to_rgb().to_ndarray() if to_rgb else frame.to_ndarray()
        for frame in video_frames
    ]
    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
        if vframes.ndim == 3:
            # grayscale video, add channel dim
            vframes = vframes.unsqueeze(-1)
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            audio_start_pts = int(floor(audio_start_pts * (1 / audio_timebase)))
            if audio_end_pts != float("inf"):
                audio_end_pts = int(ceil(audio_end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(
            aframes, audio_frames, audio_start_pts, audio_end_pts
        )
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def resample_video(
    video: torch.Tensor, original_fps: float, target_fps: float
) -> torch.Tensor:
    """
    Change the FPS of a video tensor by duplicating or removing frames.

    Args:
        video (torch.Tensor): Video tensor of shape (N, C, H, W), where N is the number of frames.
        original_fps (float): Original FPS of the video.
        target_fps (float): Target FPS of the video.

    Returns:
        torch.Tensor: Video tensor with the new FPS.
    """
    num_frames, C, H, W = video.shape

    # Calculate the scaling factor
    scale_factor = target_fps / original_fps

    if scale_factor == 1.0:
        # No change in FPS
        return video

    # Calculate the new number of frames
    new_num_frames = ceil(num_frames * scale_factor)

    # Generate indices for the new frame sequence
    if scale_factor > 1.0:
        # Duplicate frames (upsampling)
        indices = torch.arange(new_num_frames, dtype=torch.float32) / scale_factor
        indices = indices.round().long().clamp(0, num_frames - 1)
    else:
        # Remove frames (downsampling)
        indices = torch.linspace(0, num_frames - 1, new_num_frames).round().long()

    # Use the indices to gather frames
    new_video = video[indices]

    return new_video
