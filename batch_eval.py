import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import torch
import torch.distributed as distributed
import torchaudio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from saganet.data.data_setup import setup_eval_dataset
from saganet.eval_utils import ModelConfig, all_model_cfg, generate
from saganet.model.flow_matching import FlowMatching
from saganet.model.networks import MMAudio, get_my_mmaudio
from saganet.model.utils.features_utils import FeaturesUtils
from saganet.utils.video_joiner import VideoJoiner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
log = logging.getLogger()


def load_weights_in_memory(
    src_dict: dict,
    module: str,
    network: Optional[MMAudio] = None,
    features: Optional[FeaturesUtils] = None,
):
    if module == "network" and network is not None:
        if hasattr(network, "module"):
            network.module.load_weights(src_dict)  # type: ignore
        else:
            network.load_weights(src_dict)
        log.info(f"{module} weights loaded from memory.")
    if module == "synchformer" and features is not None:
        if hasattr(features.synchformer, "module"):
            features.synchformer.module.load_state_dict(src_dict, strict=True)  # type: ignore
        else:
            features.synchformer.load_state_dict(src_dict)  # type: ignore
        log.info(f"{module} weights loaded from memory.")
    if module == "lora" and network is not None:
        if hasattr(network, "module"):
            network.module.load_weights(src_dict)  # type: ignore
        else:
            network.load_weights(src_dict)
        log.info(f"{module} weights loaded from memory.")


def load_weights(
    path: str,
    module: Optional[str] = None,
    network: Optional[MMAudio] = None,
    features: Optional[FeaturesUtils] = None,
):
    # This method loads only the network weight and should be used to load a pretrained model
    per_module_sd = torch.load(path, map_location="cpu", weights_only=True)

    if module is not None:  # backwards compatibility
        per_module_sd = {module: per_module_sd}

    for module in per_module_sd:
        log.info(f"Importing {module} weights from {path}...")
        load_weights_in_memory(per_module_sd[module], module, network, features)


@torch.inference_mode()
@hydra.main(version_base="1.3.2", config_path="config", config_name="eval_config.yaml")
def main(cfg: DictConfig):
    device = "cuda"
    torch.cuda.set_device(local_rank)

    if cfg.model not in all_model_cfg:
        raise ValueError(f"Unknown model variant: {cfg.model}")
    model: ModelConfig = all_model_cfg[cfg.model]
    seq_cfg = model.seq_cfg

    run_dir = Path(HydraConfig.get().run.dir)
    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f"{cfg.dataset}-{cfg.output_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    seq_cfg.duration = cfg.duration_s
    net: MMAudio = get_my_mmaudio(
        model.model_name, use_lora=cfg.get("use_lora", False)
    ).eval()

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(
        cfg.sampling.min_sigma,
        inference_mode=cfg.sampling.method,
        num_steps=cfg.sampling.num_steps,
    )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_synchformer=True,
        enable_vae_encoder=True,
        enable_clip=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
    )
    load_weights(model.model_path, features=feature_utils, network=net)
    feature_utils = feature_utils.to(device).eval()
    net = net.to(device)
    log.info(f"Loaded weights from {model.model_path}")
    net.update_seq_lengths(
        seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
    )
    log.info(f"Latent seq len: {seq_cfg.latent_seq_len}")
    log.info(f"Clip seq len: {seq_cfg.clip_seq_len}")
    log.info(f"Sync seq len: {seq_cfg.sync_seq_len}")

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    _, loader = setup_eval_dataset(cfg.dataset, cfg)

    video_joiner = None

    with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
        for batch in tqdm(loader):
            audios = generate(
                batch.get("clip_video", None),
                batch.get("sync_video", None),
                batch.get("sync_detail_video", None),
                batch.get("caption", None),
                batch.get("mask_video", None),
                batch.get("mask_detail_video", None),
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=cfg.cfg_strength,
                clip_batch_size_multiplier=64,
                sync_batch_size_multiplier=64,
            )
            audios = audios.float().cpu()
            names = batch["name"]
            for audio, name in zip(audios, names):
                d = Path(output_dir / name).parent
                if not d.exists():
                    d.mkdir(parents=True)
                torchaudio.save(
                    output_dir / f"{name}.flac", audio, seq_cfg.sampling_rate
                )
                if video_joiner is not None:
                    video_joiner.join(
                        f"{'_'.join(name[:-4].split('_')[:-1])}",
                        name,
                        audio.transpose(0, 1),
                    )


def distributed_setup():
    distributed.init_process_group(backend="nccl")
    local_rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size


if __name__ == "__main__":
    distributed_setup()
    main()
    # clean-up
    distributed.destroy_process_group()
