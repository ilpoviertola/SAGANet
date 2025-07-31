import logging
import math
import random
from datetime import timedelta
from pathlib import Path
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from saganet.data.data_setup import setup_training_datasets, setup_val_datasets
from saganet.model.sequence_config import CONFIG_44K_SA
from saganet.runner import Runner
from saganet.sample import sample
from saganet.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from saganet.utils.logger import TensorboardLogger
from saganet.utils.synthesize_ema import synthesize_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f"Initialized: local_rank={local_rank}, world_size={world_size}")
    return local_rank, world_size


@record
@hydra.main(version_base="1.3.2", config_path="config", config_name="train_config.yaml")
def train(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # compose early such that it does not rely on future hard disk reading
    eval_cfg = compose("eval_config", overrides=[f"exp_id={cfg.exp_id}"])

    # patch data dim
    if cfg.model.endswith("44k_sa"):
        seq_cfg = CONFIG_44K_SA
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(
        cfg.exp_id,
        run_dir,
        logging.getLogger(),
        is_rank0=(local_rank == 0),
        enable_email=cfg.enable_email and not cfg.debug,
    )

    info_if_rank_zero(log, f"All configuration: {cfg}")
    info_if_rank_zero(log, f"Number of GPUs detected: {num_gpus}")

    # number of dataloader workers
    info_if_rank_zero(log, f"Number of dataloader workers (per GPU): {cfg.num_workers}")

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f"Training configuration: {cfg}")
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f"Batch size (per GPU): {cfg.batch_size}")

    # determine time to change max skip
    total_iterations = cfg["num_iterations"]

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f"Number of training samples: {len(dataset)}")
    info_if_rank_zero(log, f"Number of training batches: {len(loader)}")

    val_dataset, val_loader, eval_loader = setup_val_datasets(cfg)
    info_if_rank_zero(log, f"Number of val samples: {len(val_dataset)}")
    if val_dataset.__class__.__name__ == "ExtractedAVS":
        val_cfg = cfg.data.ExtractedAVS_val
        cfg["video_joiner_root"] = cfg.data.AVS.root
    elif val_dataset.__class__.__name__ == "MusicRaw":
        val_cfg = cfg.data.RawMusic_val
        cfg["video_joiner_root"] = cfg.data.RawMusic.root
    elif val_dataset.__class__.__name__ == "MusicHybrid":
        val_cfg = cfg.data.HybridMusic_val
        cfg["video_joiner_root"] = cfg.data.HybridMusic.root
    elif val_dataset.__class__.__name__ == "SegmentedMusicSolos":
        val_cfg = cfg.data.SMS_val
        cfg["video_joiner_root"] = cfg.data.SMS.root
    else:
        val_cfg = cfg.data.ExtractedVGG_val
        cfg["video_joiner_root"] = cfg.data.VGGSound.root

    # compute and set mean and std
    latent_mean, latent_std = dataset.compute_latent_stats()

    # construct the trainer
    trainer = Runner(
        cfg,
        log=log,
        run_path=run_dir,
        for_training=True,
        latent_mean=latent_mean,
        latent_std=latent_std,
    ).enter_train()
    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # load previous checkpoint if needed
    if cfg["checkpoint"] is not None:
        curr_iter = trainer.load_checkpoint(cfg["checkpoint"])
        cfg["checkpoint"] = None
        info_if_rank_zero(log, "Model checkpoint loaded!")
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, "Latest checkpoint loaded!")
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg["weights"] is not None:
                info_if_rank_zero(log, "Loading weights from the disk")
                trainer.load_weights(cfg["weights"], "network")
                cfg["weights"] = None

    # determine max epoch
    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log, f"We will approximately use {total_epoch} epochs.")

    # save best model
    keep_top_k_best = 3
    cur_best_val_loss = float("inf")
    best_models = OrderedDict()

    # training loop
    try:
        # Need this to select random bases in different workers
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            # Crucial for randomness!
            sampler.set_epoch(current_epoch)
            current_epoch += 1
            log.debug(f"Current epoch: {current_epoch}")

            trainer.enter_train()
            trainer.log.data_timer.start()
            for data in loader:
                trainer.train_pass(data, curr_iter)

                if (curr_iter + 1) % cfg.val_interval == 0:
                    # swap into a eval rng state, i.e., use the same seed for every validation pass
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f"Iteration {curr_iter}: validating")
                    for data in val_loader:
                        trainer.validation_pass(data, curr_iter)
                    distributed.barrier()
                    if local_rank == 0:
                        cur_val_loss = (
                            trainer.val_integrator.values["loss"]
                            / trainer.val_integrator.counts["loss"]
                        )
                        if cur_val_loss < cur_best_val_loss:
                            info_if_rank_zero(
                                log, f"Saving new best ckpt in iteration {curr_iter}."
                            )
                            cur_best_val_loss = cur_val_loss
                            model_path = trainer.save_checkpoint(
                                curr_iter, save_copy=True, saving_best=True
                            )
                            best_models[cur_val_loss] = model_path
                            if len(best_models) > keep_top_k_best:
                                best_models = OrderedDict(sorted(best_models.items()))
                                _, v = best_models.popitem(last=True)
                                v.unlink(missing_ok=True)

                    trainer.val_integrator.finalize("val", curr_iter, ignore_timer=True)
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)

                if (curr_iter + 1) % cfg.eval_interval == 0:
                    save_eval = (curr_iter + 1) % cfg.save_eval_interval == 0
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f"Iteration {curr_iter}: evaluating")
                    for data in eval_loader:
                        audio_path = trainer.inference_pass(
                            data, curr_iter, val_cfg, save_eval=save_eval
                        )
                    distributed.barrier()
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)
                    trainer.eval(audio_path, curr_iter, val_cfg)

                curr_iter += 1

                if curr_iter >= total_iterations:
                    break
    except Exception as e:
        log.error(f"Error occurred at iteration {curr_iter}!")
        log.critical(e.message if hasattr(e, "message") else str(e))
        raise
    finally:
        if not cfg.debug:
            trainer.save_checkpoint(curr_iter)
            trainer.save_weights(curr_iter)

    # Inference pass
    del trainer
    torch.cuda.empty_cache()

    # Synthesize EMA
    if local_rank == 0 and cfg.ema.enable:
        log.info(f"Synthesizing EMA with sigma={cfg.ema.default_output_sigma}")
        ema_sigma = cfg.ema.default_output_sigma
        state_dict = synthesize_ema(cfg, ema_sigma, step=None)
        save_dir = Path(run_dir) / f"{cfg.exp_id}_ema_final.pth"
        torch.save(state_dict, save_dir)
        log.info(f"Synthesized EMA saved to {save_dir}!")
    elif not cfg.ema.enable:
        log.info("EMA is not enabled, skipping EMA synthesis.")
        with open_dict(eval_cfg):
            eval_cfg.weights = Path(run_dir) / f"{cfg.exp_id}_last.pth"
    distributed.barrier()

    with open_dict(eval_cfg):
        eval_cfg.exp_id = cfg.exp_id
        eval_cfg.use_lora = cfg.get("use_lora", False)
        eval_cfg.model = cfg.model
        eval_cfg.hybrid_music_train = cfg.get("hybrid_music_train", False)
        eval_cfg.sms_train = cfg.get("sms_train", False)
    log.info(f"Evaluation: {eval_cfg}")
    sample(eval_cfg)

    # clean-up
    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    # import os
    # import debugpy

    # rank = int(os.getenv("RANK", "-1"))
    # port = rank + 5678
    # debugpy.listen(("127.0.0.1", port))
    # debugpy.wait_for_client()
    train()
