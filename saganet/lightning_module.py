from pathlib import Path
from typing import Any, Literal, Optional, Union, Callable, Tuple

import lightning
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.rank_zero import rank_zero_warn
from av_bench.evaluate import evaluate
from av_bench.extract import extract
import loralib as lora
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, MultiStepLR

from saganet.model.networks import MMAudio
from saganet.utils.video_joiner import VideoJoiner
from saganet.model.flow_matching import FlowMatching
from saganet.model.sequence_config import CONFIG_44K_SA
from saganet.model.utils.features_utils import FeaturesUtils
from saganet.model.utils.sample_utils import log_normal_sample
from saganet.model.utils.parameter_groups import get_parameter_groups


LR_SCHEDULES = Literal["constant", "poly", "step"]


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        # Network params
        network: MMAudio,
        train_network: bool,
        train_synchformer: bool,
        use_lora: bool,
        # Flow matching
        flow_matching: FlowMatching,
        # Feature extractors
        feature_utils: FeaturesUtils,
        # Sampling
        log_normal_sampling_mean: float,
        log_normal_sampling_scale: float,
        null_condition_prob: float,
        null_text_condition_prob: float,
        cfg_strength: float,
        # Video composer
        video_joiner: VideoJoiner,
        # Training params
        weight_decay: float,
        learning_rate: float,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-6,
        fused: bool = True,
        linear_warmup_steps: int = 2500,
        lr_schedule: LR_SCHEDULES = "step",
        lr_schedule_steps: Union[list[int], None] = [60_000, 65_000],
        lr_schedule_gamma: Union[float, None] = 0.1,
        evaluation_interval_epochs: int = 3,
        # Eval params
        gt_cache: Union[str, None] = None,
    ) -> None:
        super().__init__()
        assert not (
            use_lora and train_network
        ), "Cannot use LoRA when training the Synchformer"
        self.network = network
        self.train_network = train_network
        self.train_synchformer = train_synchformer
        self.use_lora = use_lora

        self.fm = flow_matching
        self.rng: Optional[torch.Generator] = None
        self.feature_utils = feature_utils.eval()

        self.log_normal_sampling_mean = log_normal_sampling_mean
        self.log_normal_sampling_scale = log_normal_sampling_scale
        self.null_condition_prob = null_condition_prob
        self.null_text_condition_prob = null_text_condition_prob
        self.cfg_strength = cfg_strength

        self.video_joiner = video_joiner

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.fused = fused
        self.linear_warmup_steps = linear_warmup_steps
        self.lr_schedule = lr_schedule
        self.lr_schedule_steps = lr_schedule_steps
        self.lr_schedule_gamma = lr_schedule_gamma
        self.evaluation_interval = evaluation_interval_epochs

        self.gt_cache = gt_cache if gt_cache is not None else ""
        self.seq_cfg = CONFIG_44K_SA

    def on_train_start(self) -> None:
        self._set_normalization_stats()
        if self.rng is None:
            self.rng = torch.Generator(self.device)

    def on_test_start(self) -> None:
        self._set_normalization_stats()
        if self.rng is None:
            self.rng = torch.Generator(self.device)

    def on_validation_start(self) -> None:
        self._set_normalization_stats()
        if self.rng is None:
            self.rng = torch.Generator(self.device)

    def on_predict_start(self) -> None:
        self._set_normalization_stats()
        if self.rng is None:
            self.rng = torch.Generator(self.device)

    def _set_normalization_stats(self) -> None:
        mean, std = self.trainer.datamodule.get_normalization_stats()  # type: ignore
        assert mean.numel() == self.network.latent_dim
        self.network.latent_mean = nn.Parameter(
            mean.view(1, 1, -1).to(self.device), requires_grad=False
        )
        self.network.latent_std = nn.Parameter(
            std.view(1, 1, -1).to(self.device), requires_grad=False
        )

    @property
    def _num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)  # type: ignore
        if self.trainer.tpu_cores:  # type: ignore
            num_devices = max(num_devices, self.trainer.tpu_cores)  # type: ignore

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs  # type: ignore

    @property
    def enable_clip(self) -> bool:
        return (
            hasattr(self.feature_utils, "clip_model")
            and self.feature_utils.clip_model is not None
        )

    @property
    def enable_synchformer(self) -> bool:
        return (
            hasattr(self.feature_utils, "synchformer")
            and self.feature_utils.synchformer is not None
        )

    @property
    def enable_vae_encoder(self) -> bool:
        # return hasattr(self.feature_utils, "tod") and self.feature_utils.tod is not None and
        return False

    def _get_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, lr_schedule: LR_SCHEDULES
    ) -> Any:
        def _warmup(currrent_step: int):
            return (currrent_step + 1) / (self.linear_warmup_steps + 1)

        wu_scheduler = LambdaLR(optimizer, lr_lambda=_warmup)

        if lr_schedule == "constant":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule == "poly":
            max_steps = self._num_training_steps
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda x: (1 - (x / max_steps)) ** 0.9,
            )
        elif lr_schedule == "step":
            assert self.lr_schedule_steps is not None
            assert self.lr_schedule_gamma is not None
            scheduler = MultiStepLR(  # type: ignore
                optimizer, self.lr_schedule_steps, self.lr_schedule_gamma
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[wu_scheduler, scheduler],
            milestones=[self.linear_warmup_steps],
        )
        return combined_scheduler

    def _get_input_feats(self, data: dict) -> dict:
        if self.enable_synchformer:
            mask_vid = data["mask_video"]
            mask_detail_vid = data["mask_detail_video"]
            sync_v = data["sync_video"]
            sync_detail_v = data["sync_detail_video"]
            sync_f = self.feature_utils.encode_video_with_sync(
                sync_v, sync_detail_v, mask_vid, mask_detail_vid
            )  # type: ignore
        else:
            sync_f = data["sync_features"]

        if self.enable_clip:
            clip_v = data["clip_video"]
            clip_f = self.feature_utils.encode_video_with_clip(clip_v).detach()  # type: ignore
            caption = data["caption"]
            text_f = self.feature_utils.encode_text(caption).detach()  # type: ignore
        else:
            clip_f = data["clip_features"]
            text_f = data["text_features"]

        if self.enable_vae_encoder:
            audio = data["audio"]
            dist = self.feature_utils.encode_audio(audio)  # type: ignore
            a_mean = dist.mean.transpose(1, 2).detach()
            a_std = dist.std.transpose(1, 2).detach()
        else:
            a_mean = data["a_mean"]
            a_std = data["a_std"]

        if "text_exist" in data:
            text_exist = data["text_exist"]
        else:
            text_exist = torch.ones(
                a_mean.shape[0], dtype=torch.bool, device=a_mean.device
            )

        if "video_exist" in data:
            video_exist = data["video_exist"]
        else:
            video_exist = torch.ones(
                a_mean.shape[0], dtype=torch.bool, device=a_mean.device
            )

        return {
            "clip_f": clip_f,
            "sync_f": sync_f,
            "text_f": text_f,
            "video_exist": video_exist,
            "text_exist": text_exist,
            "a_mean": a_mean,
            "a_std": a_std,
        }

    def _save_audios(
        self, audio: torch.Tensor, video_ids: list[str], names: list[str]
    ) -> str:
        assert audio.dim() == 3, "Audio tensor must be 3D"
        assert (
            audio.size(0) == len(video_ids) == len(names)
        ), "Batch size must match video_ids and names length"

        return ""

    def configure_optimizers(self) -> Any:
        parameter_groups = []

        if self.train_synchformer:
            self.feature_utils.requires_grad_(True)
        else:
            self.feature_utils.requires_grad_(False)

        if not self.train_network and not self.use_lora:
            self.network.requires_grad_(False)
        elif self.use_lora:
            lora.mark_only_lora_as_trainable(self.network, "lora_only")
        else:
            self.network.requires_grad_(True)

        if self.train_network or self.use_lora:
            parameter_groups.append(
                get_parameter_groups(
                    self.network,
                    weight_decay=self.weight_decay,
                    base_lr=self.learning_rate,
                )
            )
        if self.train_synchformer:
            parameter_groups.append(
                get_parameter_groups(
                    self.feature_utils,
                    weight_decay=self.weight_decay,
                    base_lr=self.learning_rate,
                )
            )
        if len(parameter_groups) == 0:
            raise ValueError("No parameters to train")

        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            fused=self.fused,
        )
        lr_scheduler = self._get_lr_scheduler(optimizer, self.lr_schedule)  # type: ignore
        return [optimizer], [lr_scheduler]

    def train_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        x1 = a_mean + a_std * a_randn
        bs = x1.shape[0]  # batch_size * seq_len * num_channels

        # normalize the latents
        x1 = self.network.normalize(x1)

        t = log_normal_sample(
            x1,
            generator=self.rng,
            m=self.log_normal_sampling_mean,
            s=self.log_normal_sampling_scale,
        )
        x0, x1, xt, (clip_f, sync_f, text_f) = self.fm.get_x0_xt_c(
            x1, t, Cs=[clip_f, sync_f, text_f], generator=self.rng
        )

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_video = samples < self.null_condition_prob
        clip_f[null_video] = self.network.empty_clip_feat
        sync_f[null_video] = self.network.empty_sync_feat

        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = samples < self.null_text_condition_prob
        text_f[null_text] = self.network.empty_string_feat

        pred_v = self.network(xt, clip_f, sync_f, text_f, t)
        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()
        return x1, loss, mean_loss, t

    def val_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        x1 = a_mean + a_std * a_randn
        bs = x1.shape[0]  # batch_size * seq_len * num_channels

        # normalize the latents
        x1 = self.network.normalize(x1)

        t = log_normal_sample(
            x1,
            generator=self.rng,
            m=self.log_normal_sampling_mean,
            s=self.log_normal_sampling_scale,
        )
        x0, x1, xt, (clip_f, sync_f, text_f) = self.fm.get_x0_xt_c(
            x1, t, Cs=[clip_f, sync_f, text_f], generator=self.rng
        )

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        # null mask is for when a video is provided but we decided to ignore it
        null_video = samples < self.null_condition_prob
        # complete mask is for when a video is not provided or we decided to ignore it
        clip_f[null_video] = self.network.empty_clip_feat
        sync_f[null_video] = self.network.empty_sync_feat

        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = samples < self.null_text_condition_prob
        text_f[null_text] = self.network.empty_string_feat

        pred_v = self.network(xt, clip_f, sync_f, text_f, t)

        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()
        return x1, loss, mean_loss, t

    def _common_step(
        self, batch: Any, fn: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_feats = self._get_input_feats(batch)
        clip_f = input_feats["clip_f"]
        sync_f = input_feats["sync_f"]
        text_f = input_feats["text_f"]
        a_mean = input_feats["a_mean"]
        a_std = input_feats["a_std"]

        # these masks are for non-existent data; masking for CFG training is in train_fn
        video_exist = input_feats["video_exist"]
        text_exist = input_feats["text_exist"]
        clip_f[~video_exist] = self.network.empty_clip_feat
        sync_f[~video_exist] = self.network.empty_sync_feat
        text_f[~text_exist] = self.network.empty_string_feat

        x1, loss, mean_loss, t = fn(clip_f, sync_f, text_f, a_mean, a_std)
        return x1, loss, mean_loss, t

    @torch.inference_mode()
    def inference_pass(self, batch: Any) -> str:
        input_feats = self._get_input_feats(batch)
        clip_f = input_feats["clip_f"]
        sync_f = input_feats["sync_f"]
        text_f = input_feats["text_f"]
        a_mean = input_feats["a_mean"]
        a_std = input_feats["a_std"]

        video_exist = input_feats["video_exist"]
        text_exist = input_feats["text_exist"]
        clip_f[~video_exist] = self.network.empty_clip_feat
        sync_f[~video_exist] = self.network.empty_sync_feat
        text_f[~text_exist] = self.network.empty_string_feat

        x0 = torch.empty_like(a_mean).normal_(generator=self.rng)
        conditions = self.network.preprocess_conditions(clip_f, sync_f, text_f)
        empty_conditions = self.network.get_empty_conditions(x0.shape[0])
        cfg_ode_wrapper = lambda t, x: self.network.ode_wrapper(
            t, x, conditions, empty_conditions, self.cfg_strength
        )
        x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
        x1_hat = self.network.unnormalize(x1_hat)
        mel = self.feature_utils.decode(x1_hat)  # type: ignore
        audio = self.feature_utils.vocode(mel).cpu()  # type: ignore

        return self._save_audios(audio, batch["id"], batch["name"])

    @rank_zero_only
    def eval_step(self, batch: Any, batch_idx: int):
        if not self.gt_cache:
            rank_zero_warn("GT cache not provided, skipping evaluation.")
            return

        _audio_dir = self.inference_pass(batch)
        audio_dir = Path(_audio_dir)
        extract(
            audio_path=audio_dir,
            output_path=audio_dir / "cache",
            device="cuda",
            batch_size=32,
            audio_length=5,
        )
        output_metrics = evaluate(
            gt_audio_cache=Path(self.gt_cache),
            pred_audio_cache=audio_dir / "cache",
        )
        self.log_dict(output_metrics, prog_bar=False, on_epoch=True, on_step=False)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x1, loss, mean_loss, t = self._common_step(batch, self.train_fn)
        return mean_loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x1, loss, mean_loss, t = self._common_step(batch, self.val_fn)
        if (self.current_epoch + 1) % self.evaluation_interval == 0:
            self.eval_step(batch, batch_idx)
        return mean_loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError("Test step is not implemented yet")

    def train(self, mode: bool = True):
        if self.train_synchformer and mode:
            self.feature_utils.synchformer.train(mode)  # type: ignore
        self.network.train(mode)
