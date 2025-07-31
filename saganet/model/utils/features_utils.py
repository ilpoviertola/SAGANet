from typing import Literal, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip import create_model_from_pretrained
from torchvision.transforms import Normalize

from saganet.ext.autoencoder import AutoEncoderModule
from saganet.ext.mel_converter import get_mel_converter
from saganet.ext.synchformer import SynchformerWithContext as Synchformer
from saganet.model.utils.distributions import DiagonalGaussianDistribution


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


class FeaturesUtils(nn.Module):

    def __init__(
        self,
        *,
        enable_synchformer: bool = True,
        enable_clip: bool = True,
        enable_vae_encoder: bool = True,
        tod_vae_ckpt: Optional[str] = None,
        bigvgan_vocoder_ckpt: Optional[str] = None,
        synchformer_ckpt: Optional[str] = None,
        mode=Literal["16k", "44k", "44k_sa"],
    ):
        super().__init__()
        self.clip_model = None
        self.synchformer = None
        self.tokenizer = None

        if enable_clip:
            self.clip_model = create_model_from_pretrained(
                "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
            )
            self.clip_preprocess = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            self.clip_model = patch_clip(self.clip_model)
            self.tokenizer = open_clip.get_tokenizer(
                "ViT-H-14-378-quickgelu"
            )  # same as 'ViT-H-14'

        if enable_synchformer:
            assert synchformer_ckpt is not None, "Synchformer checkpoint is required"
            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location="cpu")
            )

        self.tod = None
        self.mel_converter = None
        if tod_vae_ckpt is not None:
            self.tod = AutoEncoderModule(
                vae_ckpt_path=tod_vae_ckpt,
                vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                mode=mode,
                need_vae_encoder=enable_vae_encoder,
            )
            self.mel_converter = get_mel_converter(mode)

    def compile(self):
        if self.clip_model is not None:
            self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
            self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        if self.synchformer is not None:
            self.synchformer = torch.compile(self.synchformer)
        self.decode = torch.compile(self.decode)
        self.vocode = torch.compile(self.vocode)

    def requires_grad_(self, requires_grad=True):
        if self.clip_model is not None:
            self.clip_model.requires_grad_(False)
        if self.synchformer is not None:
            if hasattr(self.synchformer, "module"):
                self.synchformer.module.requires_grad_(requires_grad)
            else:
                self.synchformer.requires_grad_(requires_grad)
        if self.tod is not None:
            self.tod.requires_grad_(False)
        if self.mel_converter is not None:
            self.mel_converter.requires_grad_(False)
        return self

    @torch.no_grad()
    def encode_video_with_clip(
        self, x: torch.Tensor, batch_size: int = -1
    ) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        # x: (B, T, C, H, W) H/W: 384
        b, t, c, h, w = x.shape
        assert c == 3 and h == 384 and w == 384
        x = self.clip_preprocess(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            outputs.append(
                self.clip_model.encode_image(x[i : i + batch_size], normalize=True)
            )
        x = torch.cat(outputs, dim=0)
        # x = self.clip_model.encode_image(x, normalize=True)
        x = rearrange(x, "(b t) d -> b t d", b=b)
        return x

    def encode_video_with_sync(
        self,
        x: torch.Tensor,
        x_detail: torch.Tensor,
        x_mask: torch.Tensor,
        x_detail_mask: torch.Tensor,
        batch_size: int = -1,
    ) -> torch.Tensor:
        assert self.synchformer is not None, "Synchformer is not loaded"
        # x: (B, T, C, H, W) H/W: 224

        b_x, t_x, c_x, h_x, w_x = x.shape
        assert c_x == 3 and h_x == 224 and w_x == 224
        b_xd, t_xd, c_xd, h_xd, w_xd = x_detail.shape
        assert c_xd == 3 and h_xd == 224 and w_xd == 224
        b_xm, t_xm, c_xm, h_xm, w_xm = x_mask.shape
        assert c_xm == 1 and h_xm == 224 and w_xm == 224
        b_xdm, t_xdm, c_xdm, h_xdm, w_xdm = x_detail_mask.shape
        assert c_xdm == 1 and h_xdm == 224 and w_xdm == 224
        assert t_x == t_xd == t_xm == t_xdm
        assert b_x == b_xd == b_xm == b_xdm

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t_x - segment_size) // step_size + 1
        segments = []
        segments_mask = []
        segments_detail = []
        segments_detail_mask = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size : i * step_size + segment_size])
            segments_mask.append(
                x_mask[:, i * step_size : i * step_size + segment_size]
            )
            segments_detail.append(
                x_detail[:, i * step_size : i * step_size + segment_size]
            )
            segments_detail_mask.append(
                x_detail_mask[:, i * step_size : i * step_size + segment_size]
            )
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)
        x_mask = torch.stack(segments_mask, dim=1)
        x_detail = torch.stack(segments_detail, dim=1)
        x_detail_mask = torch.stack(segments_detail_mask, dim=1)

        outputs = []
        if batch_size < 0:
            batch_size = b_x
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        x_mask = rearrange(x_mask, "b s t c h w -> (b s) 1 t c h w")
        x_detail = rearrange(x_detail, "b s t c h w -> (b s) 1 t c h w")
        x_detail_mask = rearrange(x_detail_mask, "b s t c h w -> (b s) 1 t c h w")
        for i in range(0, b_x * num_segments, batch_size):
            outputs.append(
                self.synchformer(
                    x[i : i + batch_size],
                    x_mask[i : i + batch_size],
                    detail_x=x_detail[i : i + batch_size],
                    detail_mask=x_detail_mask[i : i + batch_size],
                )
            )
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b_x)
        return x

    @torch.no_grad()
    def encode_text(self, text: list[str]) -> torch.Tensor:
        assert self.clip_model is not None, "CLIP is not loaded"
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        # x: (B, L)
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.no_grad()
    def encode_audio(self, x) -> DiagonalGaussianDistribution:
        assert self.tod is not None, "VAE is not loaded"
        # x: (B * L)
        assert self.mel_converter is not None, "Mel converter is not loaded"
        mel = self.mel_converter(x)
        dist = self.tod.encode(mel)

        return dist

    @torch.no_grad()
    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.vocode(mel)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.tod is not None, "VAE is not loaded"
        return self.tod.decode(z.transpose(1, 2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
