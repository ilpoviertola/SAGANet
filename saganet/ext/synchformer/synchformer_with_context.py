import typing as tp

import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torchvision.ops import masks_to_boxes
from torchvision.transforms.v2 import Resize
from transformers.activations import ACT2FN

from saganet.ext.synchformer import vit_helper
from saganet.ext.synchformer.motionformer import MotionFormer
from saganet.model.transformer_layers import attention


class CrossAttentionLayer(nn.Module):
    """Multi-headed cross-attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(
        self, hidden_size: int, num_attention_heads: int, attention_dropout_p: float
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout_p = attention_dropout_p

        # self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv = nn.Linear(self.embed_dim, self.embed_dim * 2)
        self.q_norm = nn.RMSNorm(self.embed_dim // self.num_heads)
        self.k_norm = nn.RMSNorm(self.embed_dim // self.num_heads)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.split_into_heads = Rearrange(
            "b n (h d j) -> b h n d j",
            h=self.num_heads,
            d=self.embed_dim // self.num_heads,
            j=3,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[
        torch.Tensor, tp.Optional[torch.Tensor], tp.Optional[tp.Tuple[torch.Tensor]]
    ]:
        """Input shape: Batch x Time x Channel"""

        # batch_size, q_len, _ = hidden_states.size()
        # batch_size, kv_len, _ = encoder_hidden_states.size()

        # query_states: torch.Tensor = self.q_proj(hidden_states)
        # key_states: torch.Tensor = self.k_proj(encoder_hidden_states)
        # value_states: torch.Tensor = self.v_proj(encoder_hidden_states)

        # query_states = query_states.view(
        #     batch_size, q_len, self.num_heads, self.head_dim
        # ).transpose(1, 2)
        # key_states = key_states.view(
        #     batch_size, kv_len, self.num_heads, self.head_dim
        # ).transpose(1, 2)
        # value_states = value_states.view(
        #     batch_size, kv_len, self.num_heads, self.head_dim
        # ).transpose(1, 2)

        # k_v_seq_len = key_states.shape[-2]
        # attn_weights = (
        #     torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        # )

        # if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights + attention_mask

        # attn_weights = nn.functional.softmax(
        #     attn_weights, dim=-1, dtype=attn_weights.dtype
        # ).to(query_states.dtype)

        # attn_weights = nn.functional.dropout(
        #     attn_weights, p=self.dropout_p, training=self.training
        # )
        # attn_output = torch.matmul(attn_weights, value_states)
        q = self.q(hidden_states)
        kv = self.kv(encoder_hidden_states)
        qkv = torch.cat((q, kv), dim=-1)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1) * self.scale
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_output = attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_p
        )

        # if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output  # , attn_weights


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CrossAttnEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_attention_heads: int,
        attention_dropout_p: float,
        residual_dropout_p: float,
        zero_init_output: bool,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.cross_attn = CrossAttentionLayer(
            hidden_size, num_attention_heads, attention_dropout_p
        )
        self.residual_dropout = nn.Dropout(residual_dropout_p)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)

        if zero_init_output:
            self.register_parameter(
                "attn_factor", nn.Parameter(torch.zeros((1,)).view(()))
            )
            self.register_parameter(
                "mlp_factor", nn.Parameter(torch.zeros((1,)).view(()))
            )
        else:
            self.attn_factor = 1.0
            self.mlp_factor = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        # Dropping the residual: let the model leverage more on the context
        hidden_states = (
            self.residual_dropout(residual) + self.attn_factor * hidden_states
        )

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_factor * hidden_states

        return hidden_states


class MotionFormerWithContext(MotionFormer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 3D Patch Embedding
        self.patch_embed_3d_mask = vit_helper.PatchEmbed3D(
            img_size=self.img_size,
            temporal_resolution=self.temporal_resolution,
            patch_size=self.patch_size,
            in_chans=1,
            embed_dim=self.embed_dim,
            z_block_size=self.cfg.VIT.PATCH_SIZE_TEMP,
        )
        self.patch_embed_3d_mask.proj.weight.data = torch.zeros_like(
            self.patch_embed_3d_mask.proj.weight.data
        )

        # Cross-attention layers for merging of full and detail features
        # TODO: Add support for configurability of the CA-layers
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttnEncoderLayer(
                    hidden_size=self.embed_dim,
                    intermediate_size=self.embed_dim * 4,
                    num_attention_heads=12,
                    attention_dropout_p=0.0,
                    residual_dropout_p=0.0,
                    zero_init_output=True,
                    layer_norm_eps=1e-6,
                    hidden_act="gelu_pytorch_tanh",
                )
                for _ in range(len(self.blocks))
            ]
        )

        self.requires_grad_(True)

    def requires_grad_(self, requires_grad=True):
        """Set requires_grad for all parameters cross-attention layers."""
        for name, param in self.named_parameters():
            if "cross_attn_layers" in name or "patch_embed_3d_mask" in name:
                param.requires_grad = requires_grad
            else:
                param.requires_grad = False

    def forward(  # type: ignore
        self,
        full_frame: torch.Tensor,
        full_mask: torch.Tensor,
        cropped_frame: torch.Tensor,
        cropped_mask: torch.Tensor,
    ):
        """
        Args:
            full_frame: (B, S, C, T, H, W)
            full_mask: (B, S, T)
            cropped_frame: (B, S, C, T, H, W) aka. detail_x
            cropped_mask: (B, S, T) aka. detail_mask
        """
        assert full_frame.shape == cropped_frame.shape
        B, S, C, T, H, W = full_frame.shape
        orig_shape = (B, S, C, T, H, W)

        # flatten batch and segments
        full_frame = full_frame.view(B * S, C, T, H, W)
        full_mask = full_mask.view(B * S, 1, T, H, W)
        cropped_frame = cropped_frame.view(B * S, C, T, H, W)
        cropped_mask = cropped_mask.view(B * S, 1, T, H, W)
        features = self.forward_backbone(
            full_frame, full_mask, cropped_frame, cropped_mask, orig_shape=orig_shape
        )
        # unpack the segments (using rest dimensions to support different shapes e.g. (BS, D) or (BS, t, D))
        features = features.view(B, S, *features.shape[1:])
        # x is now of shape (B*S, D) or (B*S, t, D) if `self.temp_attn_agg` is `Identity`

        return features  # x is (B, S, ...)

    def forward_backbone(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cropped_x: torch.Tensor,
        cropped_x_mask: torch.Tensor,
        orig_shape: tuple,
    ) -> torch.Tensor:
        """x is of shape (1, BS, C, T, H, W) where S is the number of segments."""
        assert self.extract_features
        x = self.forward_segments(x, x_mask, orig_shape)
        x = self.forward_segments(cropped_x, cropped_x_mask, orig_shape, x)
        x = x[:, 1:, :]  # remove CLS token
        x = self.norm(x)
        x = self.pre_logits(x)

        # TODO: Spatial aggregation here or in the forward_features?
        if self.factorize_space_time:
            x = self.restore_spatio_temp_dims(
                x, orig_shape
            )  # (B*S, D, t, h, w) <- (B*S, t*h*w, D)

            x = self.spatial_attn_agg(x, None)  # (B*S, t, D)
            x = self.temp_attn_agg(
                x
            )  # (B*S, D) or (BS, t, D) if `self.temp_attn_agg` is `Identity`
        return x

    def forward_segments(  # type: ignore
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        orig_shape: tuple,
        ca_kv_features: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x is of shape (1, BS, C, T, H, W) where S is the number of segments."""
        assert self.extract_features
        x, _ = self.forward_features(x, x_mask, ca_kv_features)
        return x

    def forward_features(self, x, x_mask, ca_kv_features=None):
        B = x.shape[0]
        merge = ca_kv_features is not None

        # apply patching on input
        x = self.patch_embed_3d(x)
        x_mask = self.patch_embed_3d_mask(x_mask)
        x = x + x_mask
        tok_mask = None

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        new_pos_embed = self.pos_embed
        npatch = self.patch_embed.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if self.cfg.VIT.POS_EMBED == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(
                    1, self.temporal_resolution, 1
                )
                tile_temporal_embed = self.temp_embed.repeat_interleave(npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            elif self.cfg.VIT.POS_EMBED == "joint":
                x = x + self.st_embed
        else:
            # image input
            x = x + new_pos_embed

        # Apply positional dropout
        x = self.pos_drop(x)

        # Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            if not merge:
                with torch.no_grad():
                    x = blk(
                        x,
                        seq_len=npatch,
                        num_frames=self.temporal_resolution,
                        approx=self.cfg.VIT.APPROX_ATTN_TYPE,
                        num_landmarks=self.cfg.VIT.APPROX_ATTN_DIM,
                        tok_mask=tok_mask,
                    )
            # if merge and i < len(self.cross_attn_layers):
            else:
                x = blk(
                    x,
                    seq_len=npatch,
                    num_frames=self.temporal_resolution,
                    approx=self.cfg.VIT.APPROX_ATTN_TYPE,
                    num_landmarks=self.cfg.VIT.APPROX_ATTN_DIM,
                    tok_mask=tok_mask,
                )
                # Cross-attention layer for merging of full and detail features
                x = self.cross_attn_layers[i](x, ca_kv_features, attention_mask=None)

        return x, tok_mask


class SynchformerWithContext(nn.Module):
    MIN_CROP_H = 56
    MIN_CROP_W = 56

    def __init__(self):
        super().__init__()
        self.vfeat_extractor = MotionFormerWithContext(
            extract_features=True,
            factorize_space_time=True,
            agg_space_module="TransformerEncoderLayer",
            agg_time_module="torch.nn.Identity",
            add_global_repr=False,
        )

    def requires_grad_(self, requires_grad=True):
        self.vfeat_extractor.requires_grad_(requires_grad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs):  # type: ignore
        B, S, Tv, C, H, W = x.shape
        x = x.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        mask = mask.permute(0, 1, 3, 2, 4, 5)  # (B, S, C=1, Tv, H, W)

        if "detail_x" in kwargs and "detail_mask" in kwargs:
            detail_x, detail_mask = kwargs["detail_x"], kwargs["detail_mask"]
        else:
            detail_x, detail_mask = self._focal_crop(x, mask)
        detail_x = detail_x.permute(0, 1, 3, 2, 4, 5)
        detail_mask = detail_mask.permute(0, 1, 3, 2, 4, 5)

        return self.vfeat_extractor(x, mask, detail_x, detail_mask)

    def load_state_dict(self, sd: tp.Mapping[str, tp.Any], strict: bool = False):  # type: ignore
        return super().load_state_dict(sd, strict)

    def _focal_crop(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # support various input shapes
        if len(mask.shape) == 6:  # (B, S, C=1, T, H, W)
            B, S, C, T, H, W = mask.shape
            assert C == 1, f"{C=}"
            # mask = mask.squeeze(2)
        elif len(mask.shape) == 5:  # (B, S, T, H, W)
            B, S, T, H, W = mask.shape
            mask = mask.unsqueeze(2)
        else:
            raise ValueError(f"{mask.shape=}")
        # mask = einops.rearrange(mask, "B S T H W -> B (S T) H W")

        if isinstance(self.vfeat_extractor.img_size, int):
            img_size = (self.vfeat_extractor.img_size, self.vfeat_extractor.img_size)
        elif len(self.vfeat_extractor.img_size) == 1:
            img_size = (
                self.vfeat_extractor.img_size[0],
                self.vfeat_extractor.img_size[0],
            )
        else:
            img_size = self.vfeat_extractor.img_size
        resize = Resize(img_size, antialias=True)
        detail_x = torch.empty_like(x)
        detail_mask = torch.empty_like(mask)

        # extract the bounding boxes per batch
        for b in range(B):
            cur_mask = einops.rearrange(mask[b], "T C H W -> (T C) H W")
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
                torch.clamp(bbox[..., 2] - bbox[..., 0], min=self.MIN_CROP_W).max(),
                torch.clamp(bbox[..., 3] - bbox[..., 1], min=self.MIN_CROP_H).max(),
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
            x1 = x1 - x1_res
            y1_res = torch.where(y1 > H, y1 - H, torch.zeros_like(y1))
            y1 = y1 - y1_res

            x0 = x0 - x1_res
            y0 = y0 - y1_res

            assert x0.dim() == x1.dim() == y0.dim() == y1.dim() == 1
            assert x0.min() >= 0 and x1.max() <= W
            assert y0.min() >= 0 and y1.max() <= H
            i = 0
            for t in range(T):
                im = x[b, t, :, y0[i] : y1[i], x0[i] : x1[i]]
                im = resize(im)
                detail_x[b, t, :, :, :] = im
                m = mask[b, t, :, y0[i] : y1[i], x0[i] : x1[i]]
                m = resize(m)
                detail_mask[b, t, :, :, :] = m
                i += 1

        return detail_x, detail_mask
