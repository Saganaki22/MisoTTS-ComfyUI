"""Miso TTS model reconstruction.

The tensor names and sizes match the public MisoLabs/MisoTTS safetensors.
The generation method mirrors SesameAILabs/csm's public Model.generate_frame.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger("MisoTTS")


@dataclass(frozen=True)
class TransformerConfig:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    intermediate_dim: int
    max_seq_len: int = 2048
    norm_eps: float = 1e-5
    rope_base: int = 500_000
    rope_scale_factor: int = 32

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass(frozen=True)
class MisoModelConfig:
    text_vocab_size: int = 128_256
    audio_vocab_size: int = 2_051
    audio_num_codebooks: int = 32
    max_seq_len: int = 2048
    backbone: TransformerConfig = TransformerConfig(
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=2048,
    )
    decoder: TransformerConfig = TransformerConfig(
        num_layers=8,
        num_heads=24,
        num_kv_heads=6,
        embed_dim=1536,
        intermediate_dim=6912,
        max_seq_len=32,
    )


def create_miso_config(max_seq_len: int = 2048) -> MisoModelConfig:
    return MisoModelConfig(
        max_seq_len=max_seq_len,
        backbone=TransformerConfig(
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            embed_dim=4096,
            intermediate_dim=14336,
            max_seq_len=max_seq_len,
        ),
        decoder=TransformerConfig(
            num_layers=8,
            num_heads=24,
            num_kv_heads=6,
            embed_dim=1536,
            intermediate_dim=6912,
            max_seq_len=32,
        ),
    )


def _load_torchtune_llama3_2():
    try:
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            from torchtune.models import llama3_2
        output = stdout.getvalue().strip()
        if output and output != "import error: No module named 'triton'":
            logger.info("torchtune import output: %s", output)
        return llama3_2
    except ImportError as exc:
        raise ImportError(
            "The official Miso backend requires torchtune==0.4.0. Install it into the ComfyUI venv."
        ) from exc


def _prepare_torchtune_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _torchtune_llama3_2_8b(max_seq_len: int):
    llama3_2 = _load_torchtune_llama3_2()
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=max_seq_len,
        intermediate_dim=14_336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def _torchtune_llama3_2_300m(max_seq_len: int):
    llama3_2 = _load_torchtune_llama3_2()
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=8,
        num_heads=24,
        num_kv_heads=6,
        embed_dim=1536,
        max_seq_len=max_seq_len,
        intermediate_dim=6912,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, *, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm.to(input_dtype) * self.scale).to(input_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: TransformerConfig, *, dtype=None, device=None):
        super().__init__()
        self.w1 = nn.Linear(cfg.embed_dim, cfg.intermediate_dim, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(cfg.intermediate_dim, cfg.embed_dim, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(cfg.embed_dim, cfg.intermediate_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Llama3ScaledRoPE(nn.Module):
    """Llama 3.1/3.2 scaled RoPE, adapted from torchtune's implementation."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: int = 500_000,
        scale_factor: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scale_factor = scale_factor
        self.low_freq_factor = 1
        self.high_freq_factor = 4
        self.old_context_len = 8192
        self.register_buffer("cache", torch.empty(0), persistent=False)

    def build_cache(self, device: torch.device) -> None:
        freqs = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        theta = self._apply_scaling(freqs)
        seq_idx = torch.arange(self.max_seq_len, dtype=theta.dtype, device=device)
        idx_theta = torch.einsum("i,j->ij", seq_idx, theta).float()
        self.cache = torch.stack([idx_theta.cos(), idx_theta.sin()], dim=-1)

    def _apply_scaling(self, freqs: torch.Tensor) -> torch.Tensor:
        low_freq_wavelen = self.old_context_len / self.low_freq_factor
        high_freq_wavelen = self.old_context_len / self.high_freq_factor
        scaled = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                scaled.append(freq)
            elif wavelen > low_freq_wavelen:
                scaled.append(freq / self.scale_factor)
            else:
                smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                    self.high_freq_factor - self.low_freq_factor
                )
                scaled.append((1 - smooth) * freq / self.scale_factor + smooth * freq)
        return torch.stack(scaled)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cache.numel() == 0 or self.cache.device != x.device:
            self.build_cache(x.device)

        seq_len = x.size(1)
        if input_pos is None:
            rope_cache = self.cache[:seq_len]
        else:
            rope_cache = self.cache[input_pos]

        x_shape = x.float().reshape(*x.shape[:-1], -1, 2)
        if input_pos is None:
            rope_cache = rope_cache.view(1, seq_len, 1, x_shape.size(3), 2)
        else:
            rope_cache = rope_cache.view(x_shape.size(0), seq_len, 1, x_shape.size(3), 2)

        x_out = torch.stack(
            [
                x_shape[..., 0] * rope_cache[..., 0] - x_shape[..., 1] * rope_cache[..., 1],
                x_shape[..., 1] * rope_cache[..., 0] + x_shape[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        return x_out.flatten(3).type_as(x)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, :, :], -torch.finfo(scores.dtype).max)
    probs = scores.softmax(dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def _sage_attention_or_none(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    try:
        from sageattention import sageattn
    except Exception:
        return None

    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        return None

    try:
        is_full_causal_pass = q.shape[-2] == k.shape[-2]
        out = sageattn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            is_causal=is_full_causal_pass,
            tensor_layout="HND",
        )
        return out.contiguous()
    except Exception as exc:
        logger.warning("SageAttention failed, falling back to SDPA/eager: %s", exc)
        return None


def _flash_attention_or_none(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Optional[torch.Tensor]:
    try:
        from flash_attn import flash_attn_func
    except Exception:
        return None

    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        return None

    try:
        is_full_causal_pass = q.shape[-2] == k.shape[-2]
        out = flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            causal=is_full_causal_pass,
        )
        return out.transpose(1, 2).contiguous()
    except Exception as exc:
        logger.warning("FlashAttention failed, falling back to SDPA/eager: %s", exc)
        return None


class MisoAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig, attention_backend: str, *, dtype=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.attention_backend = attention_backend
        self.q_proj = nn.Linear(cfg.embed_dim, cfg.num_heads * cfg.head_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.num_kv_heads * cfg.head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(cfg.embed_dim, cfg.num_kv_heads * cfg.head_dim, bias=False, dtype=dtype, device=device)
        self.output_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.embed_dim, bias=False, dtype=dtype, device=device)
        self.rope = Llama3ScaledRoPE(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.rope_base,
            cfg.rope_scale_factor,
        )
        self.register_buffer("k_cache", torch.empty(0), persistent=False)
        self.register_buffer("v_cache", torch.empty(0), persistent=False)

    def setup_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        shape = (batch_size, self.cfg.num_kv_heads, self.cfg.max_seq_len, self.cfg.head_dim)
        self.k_cache = torch.empty(shape, dtype=dtype, device=device)
        self.v_cache = torch.empty(shape, dtype=dtype, device=device)
        self.rope.build_cache(device)

    def reset_cache(self) -> None:
        # The active positions are controlled by input_pos, so zeroing is not needed.
        pass

    def caches_are_enabled(self) -> bool:
        return self.k_cache.numel() > 0 and self.v_cache.numel() > 0

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.cfg.num_heads, self.cfg.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.cfg.num_kv_heads, self.cfg.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.cfg.num_kv_heads, self.cfg.head_dim)

        q = self.rope(q, input_pos)
        k = self.rope(k, input_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.caches_are_enabled() and input_pos is not None:
            if bsz != 1:
                for batch_idx in range(bsz):
                    self.k_cache[batch_idx, :, input_pos[batch_idx], :] = k[batch_idx]
                    self.v_cache[batch_idx, :, input_pos[batch_idx], :] = v[batch_idx]
            else:
                pos = input_pos[0]
                self.k_cache[:bsz, :, pos, :] = k
                self.v_cache[:bsz, :, pos, :] = v

            end_pos = int(input_pos.max().item()) + 1
            k = self.k_cache[:bsz, :, :end_pos, :]
            v = self.v_cache[:bsz, :, :end_pos, :]
            if mask is not None:
                mask = mask[:, :, :end_pos]

        n_rep = self.cfg.num_heads // self.cfg.num_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

        out = None
        if self.attention_backend == "sageattention":
            out = _sage_attention_or_none(q, k, v, mask)
        elif self.attention_backend in {"flash_attention", "flashattention"}:
            out = _flash_attention_or_none(q, k, v)

        if out is None and self.attention_backend != "eager":
            try:
                attn_mask = mask[:, None, :, :] if mask is not None else None
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            except Exception as exc:
                logger.warning("SDPA failed, falling back to eager attention: %s", exc)
                out = None

        if out is None:
            out = _eager_attention(q, k, v, mask)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.cfg.num_heads * self.cfg.head_dim)
        return self.output_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig, attention_backend: str, *, dtype=None, device=None):
        super().__init__()
        self.attn = MisoAttention(cfg, attention_backend, dtype=dtype, device=device)
        self.mlp = SwiGLUMLP(cfg, dtype=dtype, device=device)
        self.sa_norm = RMSNorm(cfg.embed_dim, cfg.norm_eps, dtype=dtype, device=device)
        self.mlp_norm = RMSNorm(cfg.embed_dim, cfg.norm_eps, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attn(self.sa_norm(x), input_pos, mask)
        return h + self.mlp(self.mlp_norm(h))


class TransformerStack(nn.Module):
    def __init__(self, cfg: TransformerConfig, attention_backend: str, *, dtype=None, device=None):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg, attention_backend, dtype=dtype, device=device) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.embed_dim, cfg.norm_eps, dtype=dtype, device=device)
        self.max_seq_len = cfg.max_seq_len

    def setup_caches(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.setup_cache(batch_size, dtype, device)

    def reset_caches(self) -> None:
        for layer in self.layers:
            layer.attn.reset_cache()

    def caches_are_enabled(self) -> bool:
        return all(layer.attn.caches_are_enabled() for layer in self.layers)

    def set_attention_backend(self, backend: str) -> None:
        for layer in self.layers:
            layer.attn.attention_backend = backend

    def build_rope_cache(self, device: torch.device) -> None:
        for layer in self.layers:
            layer.attn.rope.build_cache(device)

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, input_pos, mask)
        return self.norm(x)


def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    return mask[input_pos, :]


def _sample_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True).to(dtype=torch.long)

    logits = logits / max(temperature, 1e-6)
    vocab = logits.shape[-1]
    top_k = max(1, min(int(top_k), vocab))

    kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    logits = logits.masked_fill(logits < kth, -float("inf"))

    probs = F.softmax(logits.float(), dim=-1)
    noise = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / noise, dim=-1, keepdim=True).to(dtype=torch.long)


def _backend_attention_call(attention_backend: str):
    def attention_call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        out = None
        # Torchtune's KV-cache path passes a mask over the full cache buffer.
        # flash_attn_func/sageattn here do not consume that arbitrary mask, so
        # using them would let tokens attend to unwritten cache positions.
        can_use_unmasked_kernel = mask is None
        if can_use_unmasked_kernel and attention_backend == "sageattention":
            out = _sage_attention_or_none(q, k, v, mask)
        elif can_use_unmasked_kernel and attention_backend in {"flash_attention", "flashattention"}:
            out = _flash_attention_or_none(q, k, v)

        if out is not None:
            return out

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, :, :] if mask.ndim == 3 else mask

        try:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal and attn_mask is None,
            )
        except Exception as exc:
            logger.warning("SDPA failed, falling back to eager attention: %s", exc)
            return _eager_attention(q, k, v, mask if isinstance(mask, torch.Tensor) and mask.ndim == 3 else None)

    return attention_call


class MisoTTSModel(nn.Module):
    def __init__(self, config: MisoModelConfig, attention_backend: str = "sdpa", *, dtype=None, device=None):
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend
        self.backbone = TransformerStack(config.backbone, attention_backend, dtype=dtype, device=device)
        self.decoder = TransformerStack(config.decoder, attention_backend, dtype=dtype, device=device)
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.backbone.embed_dim, dtype=dtype, device=device)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks,
            config.backbone.embed_dim,
            dtype=dtype,
            device=device,
        )
        self.projection = nn.Linear(config.backbone.embed_dim, config.decoder.embed_dim, bias=False, dtype=dtype, device=device)
        self.codebook0_head = nn.Linear(config.backbone.embed_dim, config.audio_vocab_size, bias=False, dtype=dtype, device=device)
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1,
                config.decoder.embed_dim,
                config.audio_vocab_size,
                dtype=dtype,
                device=device,
            )
        )
        self.device = torch.device("cpu")

    def build_rope_cache(self, device: torch.device) -> None:
        self.backbone.build_rope_cache(device)
        self.decoder.build_rope_cache(device)

    def setup_caches(self, max_batch_size: int = 1) -> None:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        self.backbone.setup_caches(max_batch_size, dtype, device)
        self.decoder.setup_caches(max_batch_size, dtype, device)
        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device), persistent=False)

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend
        self.backbone.set_attention_backend(backend)
        self.decoder.set_attention_backend(backend)

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        codebook_offsets = self.config.audio_vocab_size * torch.arange(
            self.config.audio_num_codebooks,
            device=tokens.device,
        )
        audio_tokens = tokens[:, :, :-1] + codebook_offsets
        audio_embeds = self.audio_embeddings(audio_tokens.reshape(-1)).reshape(
            tokens.size(0),
            tokens.size(1),
            self.config.audio_num_codebooks,
            -1,
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"

        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)
        last_h = h[:, -1, :]

        c0_logits = self.codebook0_head(last_h)
        c0_sample = _sample_logits(c0_logits, temperature, top_k)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        self.decoder.reset_caches()
        for codebook in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h),
                input_pos=curr_pos,
                mask=curr_decoder_mask,
            ).to(dtype=dtype)
            logits = torch.mm(decoder_h[:, -1, :], self.audio_head[codebook - 1])
            ci_sample = _sample_logits(logits, temperature, top_k)
            curr_h = self._embed_audio(codebook, ci_sample)
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample


class TorchtuneMisoTTSModel(nn.Module):
    """Official MisoLabsAI/MisoTTS model layout backed by torchtune Llama3.2 blocks."""

    def __init__(self, config: MisoModelConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend
        self.backbone, backbone_dim = _prepare_torchtune_transformer(
            _torchtune_llama3_2_8b(config.max_seq_len)
        )
        self.decoder, decoder_dim = _prepare_torchtune_transformer(
            _torchtune_llama3_2_300m(config.audio_num_codebooks)
        )
        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        self.device = torch.device("cpu")
        self.set_attention_backend(attention_backend)

    def build_rope_cache(self, device: torch.device) -> None:
        device = torch.device(device)
        for module in self.modules():
            rope_init = getattr(module, "rope_init", None)
            if callable(rope_init):
                rope_init()
                module.to(device)

    @staticmethod
    def _iter_torchtune_attn_modules(stack: nn.Module):
        for module in stack.modules():
            if hasattr(module, "kv_cache") and hasattr(module, "cache_enabled"):
                yield module

    @classmethod
    def _clear_torchtune_caches(cls, stack: nn.Module) -> None:
        for module in cls._iter_torchtune_attn_modules(stack):
            module.kv_cache = None
            module.cache_enabled = False

    @classmethod
    def _torchtune_caches_ready(
        cls,
        stack: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_seq_len: int,
    ) -> bool:
        found = False
        for module in cls._iter_torchtune_attn_modules(stack):
            found = True
            kv_cache = getattr(module, "kv_cache", None)
            if kv_cache is None or not getattr(module, "cache_enabled", False):
                return False
            k_cache = getattr(kv_cache, "k_cache", None)
            v_cache = getattr(kv_cache, "v_cache", None)
            if k_cache is None or v_cache is None:
                return False
            if k_cache.is_meta or v_cache.is_meta:
                return False
            if k_cache.device != device or v_cache.device != device:
                return False
            if k_cache.dtype != dtype or v_cache.dtype != dtype:
                return False
            if k_cache.shape[2] < max_seq_len or v_cache.shape[2] < max_seq_len:
                return False
        return found

    def setup_caches(self, max_batch_size: int = 1) -> None:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        self.build_rope_cache(device)

        if not self._torchtune_caches_ready(self.backbone, device, dtype, self.backbone.max_seq_len):
            self._clear_torchtune_caches(self.backbone)
            self.backbone.setup_caches(max_batch_size, dtype)
            self.backbone.to(device)

        if not self._torchtune_caches_ready(self.decoder, device, dtype, self.config.audio_num_codebooks):
            self._clear_torchtune_caches(self.decoder)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)
            self.decoder.to(device)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device), persistent=False)
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device), persistent=False)

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def set_attention_backend(self, backend: str) -> None:
        if backend in {"flash_attention", "flashattention", "sageattention"}:
            logger.info(
                "Official torchtune backend requires mask-aware attention during KV-cache inference; using sdpa."
            )
            backend = "sdpa"
        self.attention_backend = backend
        call = _backend_attention_call(backend)
        for module in self.modules():
            if hasattr(module, "_attention_call"):
                module._attention_call = call

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)
        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.reshape(-1)).reshape(
            tokens.size(0),
            tokens.size(1),
            self.config.audio_num_codebooks,
            -1,
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"

        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = _sample_logits(c0_logits, temperature, top_k)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        self.decoder.reset_caches()
        for codebook in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h),
                input_pos=curr_pos,
                mask=curr_decoder_mask,
            ).to(dtype=dtype)
            logits = torch.mm(decoder_h[:, -1, :], self.audio_head[codebook - 1])
            ci_sample = _sample_logits(logits, temperature, top_k)
            ci_embed = self._embed_audio(codebook, ci_sample)
            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample


def load_safetensors_into_model(
    model: torch.nn.Module,
    weights_path: str | Path,
) -> tuple[list[str], list[str]]:
    from safetensors import safe_open

    weights_path = Path(weights_path)
    params = dict(model.named_parameters())
    missing: list[str] = []

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        unexpected = sorted(keys - set(params.keys()))
        for name, param in params.items():
            if name not in keys:
                missing.append(name)
                continue
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} != model {tuple(param.shape)}"
                )
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

    if missing:
        logger.warning("Missing %d MisoTTS tensors, first few: %s", len(missing), missing[:8])
    if unexpected:
        logger.warning("Unexpected %d MisoTTS tensors, first few: %s", len(unexpected), unexpected[:8])

    return missing, unexpected


def _hf_rope_parameters(rope: str) -> dict:
    if rope == "default":
        return {"rope_type": "default", "rope_theta": 500_000.0}
    if rope == "llama3_scaled":
        return {
            "rope_type": "llama3",
            "rope_theta": 500_000.0,
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    raise ValueError(f"Unsupported RoPE mode: {rope}")


def _hf_attention_name(attention_backend: str) -> str:
    if attention_backend in {"flash_attention", "flashattention"}:
        return "flash_attention_2"
    if attention_backend in {"sdpa", "sageattention"}:
        return "sdpa"
    return "sdpa"


class HFCSMMisoModel(nn.Module):
    """Miso wrapper backed by transformers.CsmForConditionalGeneration."""

    def __init__(
        self,
        max_seq_len: int = 2048,
        attention_backend: str = "sdpa",
        rope: str = "llama3_scaled",
    ):
        super().__init__()
        from transformers import CsmConfig, CsmForConditionalGeneration

        rope_parameters = _hf_rope_parameters(rope)
        hf_max_positions = max_seq_len
        hf_decoder_max_positions = 33
        if rope == "llama3_scaled":
            # Transformers validates llama3 RoPE as an extended-context config.
            # Miso still exposes/uses a 2048-frame context through the wrapper.
            hf_max_positions = max(max_seq_len, 8193)
            hf_decoder_max_positions = 8193
        config = CsmConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=hf_max_positions,
            rope_parameters=rope_parameters,
            attn_implementation=_hf_attention_name(attention_backend),
            depth_decoder_config={
                "backbone_hidden_size": 4096,
                "hidden_size": 1536,
                "intermediate_size": 6912,
                "num_hidden_layers": 8,
                "num_attention_heads": 24,
                "num_key_value_heads": 6,
                "head_dim": 64,
                "max_position_embeddings": hf_decoder_max_positions,
                "num_codebooks": 32,
                "vocab_size": 2051,
                "rope_parameters": rope_parameters,
                "attn_implementation": _hf_attention_name(attention_backend),
            },
        )
        self.inner = CsmForConditionalGeneration(config)
        # The Comfy runtime loads Mimi separately so it can live in models/misotts/codecs.
        self.inner.codec_model = nn.Identity()
        self.config = SimpleNamespace(
            audio_num_codebooks=32,
            audio_vocab_size=2051,
            max_seq_len=max_seq_len,
        )
        self.attention_backend = attention_backend
        self.rope = rope
        self.device = torch.device("cpu")
        self._backbone_past_key_values = None

    def build_rope_cache(self, device: torch.device) -> None:
        return None

    def setup_caches(self, max_batch_size: int = 1) -> None:
        self._backbone_past_key_values = None

    def reset_caches(self) -> None:
        self._backbone_past_key_values = None

    def set_attention_backend(self, backend: str) -> None:
        self.attention_backend = backend

    def _embed_tokens(self, tokens: torch.Tensor, tokens_mask: torch.Tensor) -> torch.Tensor:
        text_embeds = self.inner.embed_text_tokens(tokens[:, :, -1]).unsqueeze(-2)
        offsets = self.config.audio_vocab_size * torch.arange(
            self.config.audio_num_codebooks,
            device=tokens.device,
        )
        audio_tokens = tokens[:, :, :-1] + offsets
        audio_embeds = self.inner.backbone_model.embed_tokens.embed_audio_tokens(audio_tokens.reshape(-1)).reshape(
            tokens.size(0),
            tokens.size(1),
            self.config.audio_num_codebooks,
            -1,
        )
        embeds = torch.cat([audio_embeds, text_embeds], dim=-2)
        return (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        h = self._embed_tokens(tokens, tokens_mask)
        cache_position = input_pos[0].to(device=tokens.device)
        backbone_outputs = self.inner.backbone_model(
            inputs_embeds=h,
            past_key_values=self._backbone_past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        self._backbone_past_key_values = backbone_outputs.past_key_values
        last_h = backbone_outputs.last_hidden_state[:, -1, :]

        c0_logits = self.inner.lm_head(last_h)
        c0_sample = _sample_logits(c0_logits, temperature, top_k)

        decoder_input_ids = F.pad(c0_sample, (1, 0), value=0)
        do_sample = temperature > 0
        depth_outputs = self.inner.depth_decoder.generate(
            input_ids=decoder_input_ids,
            backbone_last_hidden_state=last_h.clone(),
            do_sample=do_sample,
            temperature=max(float(temperature), 1e-6) if do_sample else None,
            top_k=int(top_k) if int(top_k) > 0 else None,
            min_new_tokens=self.config.audio_num_codebooks - 1,
            max_new_tokens=self.config.audio_num_codebooks - 1,
            pad_token_id=self.config.audio_vocab_size - 1,
        )
        sequences = depth_outputs if isinstance(depth_outputs, torch.Tensor) else depth_outputs.sequences
        return sequences[:, 1:].long()


def _map_miso_key_to_hf(key: str) -> str | None:
    direct = {
        "text_embeddings.weight": "embed_text_tokens.weight",
        "audio_embeddings.weight": "backbone_model.embed_tokens.embed_audio_tokens.weight",
        "codebook0_head.weight": "lm_head.weight",
        "backbone.norm.scale": "backbone_model.norm.weight",
        "decoder.norm.scale": "depth_decoder.model.norm.weight",
        "projection.weight": "depth_decoder.model.inputs_embeds_projector.weight",
        "audio_head": "depth_decoder.codebooks_head.weight",
    }
    if key in direct:
        return direct[key]

    for prefix, dst_prefix in (
        ("backbone.layers.", "backbone_model.layers."),
        ("decoder.layers.", "depth_decoder.model.layers."),
    ):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        layer_idx, suffix = rest.split(".", 1)
        suffix_map = {
            "attn.q_proj.weight": "self_attn.q_proj.weight",
            "attn.k_proj.weight": "self_attn.k_proj.weight",
            "attn.v_proj.weight": "self_attn.v_proj.weight",
            "attn.output_proj.weight": "self_attn.o_proj.weight",
            "mlp.w1.weight": "mlp.gate_proj.weight",
            "mlp.w3.weight": "mlp.up_proj.weight",
            "mlp.w2.weight": "mlp.down_proj.weight",
            "sa_norm.scale": "input_layernorm.weight",
            "mlp_norm.scale": "post_attention_layernorm.weight",
        }
        mapped_suffix = suffix_map.get(suffix)
        if mapped_suffix is not None:
            return f"{dst_prefix}{layer_idx}.{mapped_suffix}"
    return None


def load_safetensors_into_hf_csm(
    model: HFCSMMisoModel,
    weights_path: str | Path,
) -> tuple[list[str], list[str]]:
    from safetensors import safe_open

    weights_path = Path(weights_path)
    params = dict(model.inner.named_parameters())
    loaded: set[str] = set()
    unmapped: list[str] = []

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            target_name = _map_miso_key_to_hf(name)
            if target_name is None:
                unmapped.append(name)
                continue
            if target_name not in params:
                raise RuntimeError(f"Mapped Miso tensor {name} to missing HF tensor {target_name}")
            param = params[target_name]
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name} -> {target_name}: "
                    f"checkpoint {tuple(tensor.shape)} != model {tuple(param.shape)}"
                )
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded.add(target_name)

    # Raw Sesame CSM uses the audio embedding table for the depth decoder too.
    model.inner.depth_decoder.model.embed_tokens.weight = (
        model.inner.backbone_model.embed_tokens.embed_audio_tokens.weight
    )

    missing = [
        name
        for name in params
        if not name.startswith("codec_model.")
        and name != "depth_decoder.model.embed_tokens.weight"
        and name not in loaded
    ]
    if missing:
        logger.warning("Missing %d native CSM tensors, first few: %s", len(missing), missing[:8])
    if unmapped:
        logger.warning("Unmapped %d Miso tensors, first few: %s", len(unmapped), unmapped[:8])

    return missing, unmapped
