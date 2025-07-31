import logging
import os
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
import torch.distributed.elastic
import torch.distributed.elastic.multiprocessing
import torch.distributed.elastic.multiprocessing.errors
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from saganet.data.data_setup import error_avoidance_collate
from saganet.data.extraction.segmented_music_solos import SegmentedMusicSolos
from saganet.data.extraction.urmp import URMPDataset
from saganet.model.utils.features_utils import FeaturesUtils
from saganet.utils.dist_utils import local_rank, world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# for the 44.1kHz model
"""
NOTE: 352800 (8*44100) is not divisible by (STFT hop size * VAE downsampling ratio) which is 1024.
353280 is the next integer divisible by 1024.
"""

DATASET = "urmp"  # "sms" or "urmp"
SAMPLING_RATE = 44100
DURATION_SEC = 5.0
NUM_SAMPLES = 221184
vae_path = "./ext_weights/v1-44.pth"
bigvgan_path = None
mode = "44k_sa"

synchformer_ckpt = "./ext_weights/synchformer_state_dict.pth"

# per-GPU
BATCH_SIZE = 3
NUM_WORKERS = 8

log = logging.getLogger()
log.setLevel(logging.INFO)

# uncomment the train/test/val sets to extract latents for them
# data_cfg = {
#     "train": {
#         "root": "./data/sms/data",
#         "tsv": "./data/sms/metadata.csv",
#         "normalize_audio": False,
#     },
#     "val": {
#         "root": "./data/sms/data",
#         "tsv": "./data/sms/metadata.csv",
#         "normalize_audio": False,
#     },
# }
data_cfg = {
    "test": {
        "root": "./data/urmp/data",
        "tsv": "./data/urmp/metadata.tsv",
        "normalize_audio": False,
    },
}


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size


def setup_dataset(split: str):
    if DATASET == "sms":
        dataset = SegmentedMusicSolos(  # type: ignore
            root=str(data_cfg[split]["root"]),
            split=split,
            tsv_path=str(data_cfg[split]["tsv"]),
            duration_sec=DURATION_SEC,
            sample_rate=SAMPLING_RATE,
            audio_samples=NUM_SAMPLES,
            normalize_audio=False,
        )
    elif DATASET == "urmp":
        dataset = URMPDataset(  # type: ignore
            root=str(data_cfg[split]["root"]),
            tsv_path=str(data_cfg[split]["tsv"]),
            sample_rate=SAMPLING_RATE,
            duration_sec=DURATION_SEC,
            audio_samples=NUM_SAMPLES,
            normalize_audio=False,
        )
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


@torch.distributed.elastic.multiprocessing.errors.record
@torch.inference_mode()
def extract():
    # initial setup
    distributed_setup()

    parser = ArgumentParser()
    parser.add_argument(
        "--latent_dir", type=Path, default="./training/example_output/video-latents"
    )
    parser.add_argument(
        "--output_dir", type=Path, default="./training/example_output/memmap"
    )
    args = parser.parse_args()

    latent_dir = args.latent_dir
    output_dir = args.output_dir

    # cuda setup
    torch.cuda.set_device(local_rank)
    feature_extractor = (
        FeaturesUtils(
            tod_vae_ckpt=vae_path,
            enable_synchformer=True,
            enable_clip=True,
            enable_vae_encoder=True,
            bigvgan_vocoder_ckpt=bigvgan_path,
            synchformer_ckpt=synchformer_ckpt,
            mode=mode,
        )
        .eval()
        .cuda()
    )

    for split in data_cfg.keys():
        print(f"Extracting latents for the {split} split")
        this_latent_dir = latent_dir / split
        this_latent_dir.mkdir(parents=True, exist_ok=True)

        # setup datasets
        dataset, loader = setup_dataset(split)
        log.info(f"Number of samples: {len(dataset)}")
        log.info(f"Number of batches: {len(loader)}")

        for curr_iter, data in enumerate(tqdm(loader)):
            output = {
                "id": data["id"],
                "caption": data["caption"],
            }

            audio = data["audio"].cuda()
            dist = feature_extractor.encode_audio(audio)
            output["mean"] = dist.mean.detach().cpu().transpose(1, 2)
            output["std"] = dist.std.detach().cpu().transpose(1, 2)

            clip_video = data["clip_video"].cuda()
            clip_features = feature_extractor.encode_video_with_clip(clip_video)
            output["clip_features"] = clip_features.detach().cpu()

            caption = data["caption"]
            text_features = feature_extractor.encode_text(caption)
            output["text_features"] = text_features.detach().cpu()

            torch.save(output, this_latent_dir / f"r{local_rank}_{curr_iter}.pth")

        distributed.barrier()

        # combine the results
        if local_rank == 0:
            print("Extraction done. Combining the results.")

            used_id = set()
            list_of_ids_and_labels = []
            output_data = {
                "mean": [],
                "std": [],
                "clip_features": [],
                "text_features": [],
            }

            for t in tqdm(sorted(os.listdir(this_latent_dir))):
                data = torch.load(this_latent_dir / t, weights_only=True)
                bs = len(data["id"])

                for bi in range(bs):
                    this_id = data["id"][bi]
                    this_caption = data["caption"][bi]
                    if this_id in used_id:
                        print("Duplicate id:", this_id)
                        continue

                    list_of_ids_and_labels.append(
                        {"id": this_id, "label": this_caption}
                    )
                    used_id.add(this_id)
                    output_data["mean"].append(data["mean"][bi])
                    output_data["std"].append(data["std"][bi])
                    output_data["clip_features"].append(data["clip_features"][bi])
                    output_data["text_features"].append(data["text_features"][bi])

            output_dir.mkdir(parents=True, exist_ok=True)
            output_df = pd.DataFrame(list_of_ids_and_labels)
            prefix = DATASET
            output_df.to_csv(
                output_dir / f"{prefix}-{split}.tsv", sep="\t", index=False
            )

            print(f"Output: {len(output_df)}")

            output_data = {k: torch.stack(v) for k, v in output_data.items()}
            td.TensorDict(output_data).memmap_(output_dir / f"{prefix}-{split}")


if __name__ == "__main__":
    extract()
    distributed.destroy_process_group()
