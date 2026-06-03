"""ComfyUI node definitions for Miso TTS."""

from __future__ import annotations

import logging
from typing import Tuple

import torch

from .loader import (
    ATTENTION_OPTIONS,
    DEFAULT_MIMI_CODEC,
    DEFAULT_TEXT_TOKENIZER,
    DTYPE_OPTIONS,
    get_model_choices,
    load_miso_bundle,
    resume_bundle_to_device,
)
from .runtime import (
    MisoTTSGenerator,
    comfy_audio_to_tensor,
    manual_seed_all,
    split_text_at_sentence_boundaries,
    tensor_audio_to_comfy,
)

logger = logging.getLogger("MisoTTS")
ESTIMATED_SPEECH_WORDS_PER_SECOND = 2.4

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None


class MisoTTSLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    get_model_choices(),
                    {
                        "default": "MisoTTS 8B Official (fp32)",
                        "tooltip": "Built-in MisoTTS preset. Models download to ComfyUI/models/misotts/.",
                    },
                ),
                "dtype": (
                    DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Weight precision for the Miso transformer weights.",
                    },
                ),
                "attention": (
                    ATTENTION_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention backend. sdpa uses PyTorch scaled-dot-product attention; "
                            "flash_attention and sageattention use optional CUDA kernels when available."
                        ),
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download the selected MisoTTS preset plus tokenizer and codec into ComfyUI/models/misotts/ if missing.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MISO_MODEL",)
    RETURN_NAMES = ("miso_model",)
    FUNCTION = "load"
    CATEGORY = "Miso TTS"
    DESCRIPTION = "Load Miso TTS 8B weights from ComfyUI/models/misotts/ with dtype and attention controls."

    def load(
        self,
        model: str,
        dtype: str,
        attention: str,
        download_if_missing: bool,
    ) -> Tuple[object]:
        bundle = load_miso_bundle(
            model_name=model,
            dtype_name=dtype,
            attention=attention,
            download_if_missing=download_if_missing,
            max_seq_len=2048,
            rope="llama3_scaled",
        )
        return (bundle,)


class MisoTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "miso_model": (
                    "MISO_MODEL",
                    {
                        "tooltip": "Output from Miso TTS - Load Model.",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is Miso TTS running inside ComfyUI.",
                        "tooltip": "Text to synthesize.",
                    },
                ),
                "reference_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Transcript for reference_audio. Connect Miso TTS - Whisper Transcribe here.",
                    },
                ),
                "speaker": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99,
                        "step": 1,
                        "tooltip": "Speaker tag used in the prompt, e.g. [0]. Leave 0 for normal single-speaker use; change only for multi-speaker context.",
                    },
                ),
                "max_audio_length_seconds": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.08,
                        "max": 120.0,
                        "step": 0.5,
                        "tooltip": "Maximum generated audio per chunk in seconds.",
                    },
                ),
                "longform_chunking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Split long text at sentence boundaries. Each chunk reuses the original reference audio if connected.",
                    },
                ),
                "words_per_chunk": (
                    "INT",
                    {
                        "default": 80,
                        "min": 0,
                        "max": 500,
                        "step": 10,
                        "tooltip": "Target words per longform chunk. Longform may lower this to fit max_audio_length_seconds. 0 disables text splitting.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. 0 is greedy.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 2051,
                        "step": 1,
                        "tooltip": "Top-k sampling over the 2051 audio vocabulary.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 picks a random seed.",
                    },
                ),
            },
            "optional": {
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "Optional prompt audio for voice/reference context. This is not guaranteed speaker matching.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Miso TTS"
    DESCRIPTION = "Generate ComfyUI AUDIO from text, with optional prompt audio/reference context."

    def generate(
        self,
        miso_model,
        text: str,
        reference_text: str,
        speaker: int,
        max_audio_length_seconds: float = 10.0,
        longform_chunking: bool = False,
        words_per_chunk: int = 80,
        temperature: float = 0.9,
        top_k: int = 50,
        seed: int = 0,
        reference_audio: dict | None = None,
        **kwargs,
    ) -> Tuple[dict]:
        if not text.strip():
            raise ValueError("Text cannot be empty.")

        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()

        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        manual_seed_all(int(actual_seed))

        resume_bundle_to_device(miso_model)

        generator_key = (DEFAULT_TEXT_TOKENIZER, DEFAULT_MIMI_CODEC)
        generator = miso_model.generator_cache.get(generator_key)
        if generator is None:
            generator = MisoTTSGenerator(
                miso_model,
                tokenizer_repo=DEFAULT_TEXT_TOKENIZER,
                mimi_repo=DEFAULT_MIMI_CODEC,
            )
            miso_model.generator_cache[generator_key] = generator

        ref_tensor = None
        if reference_audio is not None:
            ref_tensor = comfy_audio_to_tensor(reference_audio, generator.sample_rate)
            if not reference_text.strip():
                raise ValueError(
                    "reference_text is required when reference_audio is connected. "
                    "Connect Miso TTS - Whisper Transcribe to reference_text."
                )

        if "max_audio_length_ms" in kwargs:
            max_audio_length_ms = int(kwargs["max_audio_length_ms"])
            max_audio_length_seconds = max_audio_length_ms / 1000.0
        else:
            max_audio_length_ms = max(80, int(round(float(max_audio_length_seconds) * 1000.0)))

        frames_per_chunk = max(1, int(max_audio_length_ms / 80))
        effective_words_per_chunk = int(words_per_chunk)
        if bool(longform_chunking) and effective_words_per_chunk > 0:
            duration_word_cap = max(8, int(float(max_audio_length_seconds) * ESTIMATED_SPEECH_WORDS_PER_SECOND))
            if effective_words_per_chunk > duration_word_cap:
                logger.info(
                    "Longform chunking adjusted words_per_chunk from %d to %d for %.2fs chunk cap. "
                    "Increase max_audio_length_seconds to allow larger chunks.",
                    effective_words_per_chunk,
                    duration_word_cap,
                    float(max_audio_length_seconds),
                )
                effective_words_per_chunk = duration_word_cap

        chunks = (
            split_text_at_sentence_boundaries(text.strip(), effective_words_per_chunk)
            if bool(longform_chunking)
            else [text.strip()]
        )
        if not chunks:
            raise ValueError("Text cannot be empty.")

        total_steps = max(1, frames_per_chunk * len(chunks))
        pbar = ProgressBar(total_steps) if ProgressBar is not None else None
        progress_stride = max(1, total_steps // 10)
        completed_steps = 0
        last_logged_step = -progress_stride

        logger.info(
            "Miso TTS generate: %d chunk(s), %.2fs cap per chunk (%d Mimi frames), seed=%d.",
            len(chunks),
            float(max_audio_length_seconds),
            frames_per_chunk,
            int(actual_seed),
        )
        if len(chunks) > 1:
            logger.info("Longform chunking enabled; splitting at sentence boundaries.")
            logger.info(
                "Longform chunk word counts: %s",
                ", ".join(str(len(chunk.split())) for chunk in chunks[:24])
                + ("..." if len(chunks) > 24 else ""),
            )

        audio_chunks: list[torch.Tensor] = []

        def update_progress(current: int, total: int) -> None:
            nonlocal last_logged_step
            absolute_step = min(total_steps, completed_steps + current)
            if pbar is not None:
                pbar.update_absolute(absolute_step, total_steps)
            if absolute_step - last_logged_step >= progress_stride or current >= total:
                logger.info(
                    "Miso TTS progress: %d/%d frames (%.1fs/%.1fs).",
                    absolute_step,
                    total_steps,
                    absolute_step * 0.08,
                    total_steps * 0.08,
                )
                last_logged_step = absolute_step
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

        for chunk_idx, chunk_text in enumerate(chunks):
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

            if len(chunks) > 1:
                preview = chunk_text[:70] + ("..." if len(chunk_text) > 70 else "")
                logger.info("Miso TTS chunk %d/%d: %s", chunk_idx + 1, len(chunks), preview)

            manual_seed_all(int(actual_seed))
            context_segments: list[tuple[int, str, torch.Tensor]] = []
            if ref_tensor is not None:
                context_segments.append((int(speaker), reference_text.strip(), ref_tensor))

            audio = generator.generate(
                text=chunk_text,
                speaker=int(speaker),
                max_audio_length_ms=int(max_audio_length_ms),
                temperature=float(temperature),
                top_k=int(top_k),
                ref_audio=None if context_segments else ref_tensor,
                ref_text="" if context_segments else reference_text.strip(),
                ref_speaker=int(speaker),
                context_segments=context_segments or None,
                progress=update_progress,
            )
            audio = audio.detach().float().cpu().squeeze()
            audio_chunks.append(audio)

            completed_steps += frames_per_chunk
            if pbar is not None:
                pbar.update_absolute(min(completed_steps, total_steps), total_steps)

        audio = audio_chunks[0] if len(audio_chunks) == 1 else torch.cat(audio_chunks, dim=-1)
        logger.info("Miso TTS final audio: %.2fs at %dHz.", audio.numel() / generator.sample_rate, generator.sample_rate)
        return (tensor_audio_to_comfy(audio, generator.sample_rate),)
