import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from saganet.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    load_mask_video,
    make_video,
    setup_eval_logging,
)
from saganet.model.flow_matching import FlowMatching
from saganet.model.networks import MMAudio, get_my_mmaudio
from saganet.model.utils.features_utils import FeaturesUtils
from batch_eval import load_weights
from training.extract_focal_prompt import run_focal_crop

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="saganet_small",
        help="saganet_small, saganet_small_no_lora",
    )
    parser.add_argument("--video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--mask_video", type=Path, help="Path to the mask video file", default=""
    )
    parser.add_argument("--prompt", type=str, help="Input prompt", default="")
    parser.add_argument(
        "--negative_prompt", type=str, help="Negative prompt", default=""
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--cfg_strength", type=float, default=7.0)
    parser.add_argument("--num_steps", type=int, default=25)

    parser.add_argument("--mask_away_clip", action="store_true")

    parser.add_argument(
        "--output", type=Path, help="Output directory", default="./output"
    )
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--skip_video_composite", action="store_true")
    parser.add_argument("--full_precision", action="store_true")

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {args.variant}")
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    assert args.video, "Video path must be provided"
    video_path: Path = Path(args.video).expanduser()
    if args.mask_video:
        mask_video_path: Path = Path(args.mask_video).expanduser()
    else:
        raise NotImplementedError("On-the-fly mask generation is not implemented (yet)")

    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        log.warning("CUDA/MPS are not available, running on CPU")
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

    log.info(f"Using video {video_path}")
    video_info = load_video(video_path, duration)
    clip_frames = video_info.clip_frames
    sync_frames = video_info.sync_frames
    duration = video_info.duration_sec
    if mask_away_clip:
        clip_frames = None
    else:
        clip_frames = clip_frames.unsqueeze(0)
    sync_frames = sync_frames.unsqueeze(0)

    # TODO: add support for masked object selection
    if mask_video_path is not None:
        log.info(f"Using mask video {mask_video_path}")
        mask_video_info = load_mask_video(mask_video_path, duration)
        mask_frames = mask_video_info.sync_frames.unsqueeze(0)
    else:
        log.info("No mask video provided -- using SAM")
        raise NotImplementedError(
            "On-the-fly mask generation is not implemented (yet). Please provide a mask video"
        )

    if mask_frames is not None and sync_frames is not None:
        detail_sync, detail_mask = run_focal_crop(sync_frames, mask_frames)
    else:
        raise NotImplementedError("Please provide video and mask video")

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_synchformer=True,
        enable_clip=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        enable_vae_encoder=False,
    )
    feature_utils = feature_utils.to(device, dtype).eval()
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()

    # load a pretrained model
    load_weights(model.model_path, features=feature_utils, network=net)
    log.info(f"Loaded weights from {model.model_path}")

    seq_cfg.duration = duration
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
    )

    log.info(f"Prompt: {prompt}")
    log.info(f"Negative prompt: {negative_prompt}")

    audios = generate(
        clip_video=clip_frames,
        sync_video=sync_frames,
        sync_detail_video=detail_sync,
        text=[prompt],
        mask_video=mask_frames,
        mask_detail_video=detail_mask,
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength,
    )
    audio = audios.float().cpu()[0]
    if video_path is not None:
        save_path = output_dir / f"{video_path.stem}.flac"
    else:
        safe_filename = prompt.replace(" ", "_").replace("/", "_").replace(".", "")
        save_path = output_dir / f"{safe_filename}.flac"
    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)

    log.info(f"Audio saved to {save_path}")
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f"{video_path.stem}.mp4"
        # TODO: overlay mask on video
        make_video(
            video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate
        )
        log.info(f"Video saved to {output_dir / video_save_path}")

    log.info("Memory usage: %.2f GB", torch.cuda.max_memory_allocated() / (2**30))


if __name__ == "__main__":
    main()
