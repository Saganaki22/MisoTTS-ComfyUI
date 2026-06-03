"""Whisper ASR helpers for MisoTTS reference-audio transcription."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("MisoTTS")

warnings.filterwarnings(
    "ignore",
    message=r"A custom logits processor of type .* has been passed to `.generate\(\)`",
    category=UserWarning,
)

WHISPER_DTYPE_OPTIONS = ["auto", "fp16", "bf16", "fp32"]
WHISPER_TASK_OPTIONS = ["transcribe", "translate"]
WHISPER_LANGUAGE_OPTIONS = [
    "auto",
    "english",
    "chinese",
    "japanese",
    "korean",
    "french",
    "german",
    "spanish",
    "portuguese",
    "russian",
    "italian",
    "hindi",
    "arabic",
]

POPULAR_WHISPER_MODELS = {
    "whisper-large-v3-turbo (auto-download)": "openai/whisper-large-v3-turbo",
    "whisper-large-v3 (auto-download)": "openai/whisper-large-v3",
    "whisper-medium (auto-download)": "openai/whisper-medium",
    "whisper-small (auto-download)": "openai/whisper-small",
    "whisper-tiny (auto-download)": "openai/whisper-tiny",
}

_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def audio_encoders_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / "audio_encoders"
    except Exception:
        base = Path(__file__).resolve().parent / "models" / "audio_encoders"
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_audio_encoders_folder() -> None:
    try:
        import folder_paths

        base = str(audio_encoders_dir())
        if "audio_encoders" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("audio_encoders", base)
        logger.info("Audio encoders folder registered: %s", base)
    except Exception:
        pass


def _has_whisper_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    try:
        has_weights = any(
            item.is_file() and item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            for item in path.iterdir()
        )
    except OSError:
        return False
    return has_config and has_weights


def whisper_model_choices() -> list[str]:
    choices = list(POPULAR_WHISPER_MODELS)
    known = {_safe_repo_name(repo_id) for repo_id in POPULAR_WHISPER_MODELS.values()}
    try:
        for entry in sorted(audio_encoders_dir().iterdir()):
            if entry.is_dir() and entry.name not in known and _has_whisper_files(entry):
                choices.append(entry.name)
    except OSError:
        pass
    return choices


def _download_whisper(repo_id: str, download_if_missing: bool) -> Path:
    dest = audio_encoders_dir() / _safe_repo_name(repo_id)
    if _has_whisper_files(dest):
        return dest
    if not download_if_missing:
        raise FileNotFoundError(
            f"Whisper model is missing at {dest}. Enable download_if_missing or add it manually."
        )

    from huggingface_hub import snapshot_download

    logger.info("Downloading Whisper model %s to %s", repo_id, dest)
    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "ignore_patterns": ["*.msgpack", "*.h5", "tf_model*", "flax_model*"],
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)

    if not _has_whisper_files(dest):
        raise RuntimeError(f"Whisper download finished, but usable files were not found at {dest}.")
    return dest


def _resolve_whisper_path(model_name: str, download_if_missing: bool) -> Path:
    if model_name in POPULAR_WHISPER_MODELS:
        return _download_whisper(POPULAR_WHISPER_MODELS[model_name], download_if_missing)

    path = audio_encoders_dir() / model_name
    if _has_whisper_files(path):
        return path

    repo_id = model_name.replace("_", "/", 1)
    if "/" in repo_id:
        return _download_whisper(repo_id, download_if_missing)

    raise FileNotFoundError(
        f"Whisper model not found: {path}. Place a HuggingFace Whisper model folder under "
        "ComfyUI/models/audio_encoders/ or choose an auto-download option."
    )


def _resolve_device() -> str:
    try:
        import comfy.model_management as mm

        device = torch.device(mm.get_torch_device())
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        return f"cuda:{device.index or 0}"
    if device.type == "xpu":
        return f"xpu:{device.index or 0}"
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device.startswith("cuda"):
            try:
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                return torch.float16
        if device.startswith("xpu"):
            return torch.bfloat16
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported Whisper dtype: {dtype}")


def get_whisper_pipeline(model_name: str, dtype: str, download_if_missing: bool):
    device = _resolve_device()
    key = (model_name, dtype, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        patcher = getattr(cached, "_misotts_aimdo_patcher", None)
        if patcher is not None:
            try:
                from .loader import resume_runtime_module

                resume_runtime_module(patcher, torch.device(device))
            except Exception as exc:
                logger.warning("Could not resume Whisper ASR through ComfyUI model management: %s", exc)
        return cached

    from transformers import pipeline as hf_pipeline

    model_path = _resolve_whisper_path(model_name, download_if_missing)
    torch_dtype = _resolve_dtype(dtype, device)
    logger.info("Loading Whisper ASR from %s on %s with %s", model_path, device, torch_dtype)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        torch_dtype=torch_dtype,
        device=device,
    )
    try:
        from .loader import register_runtime_module

        patcher = register_runtime_module(pipe.model, torch.device(device))
        setattr(pipe, "_misotts_aimdo_patcher", patcher)
    except Exception as exc:
        logger.warning("Could not register Whisper ASR with ComfyUI model management: %s", exc)

    _PIPELINE_CACHE[key] = pipe
    return pipe


def comfy_audio_to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    wav = waveform[0].detach().float().cpu()
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    return wav.numpy().astype(np.float32, copy=False), sample_rate


def transcribe_audio(
    audio: dict,
    model_name: str,
    dtype: str,
    language: str,
    task: str,
    chunk_length_s: int,
    download_if_missing: bool,
) -> str:
    pipe = get_whisper_pipeline(model_name, dtype, download_if_missing)
    audio_np, sample_rate = comfy_audio_to_numpy(audio)
    generate_kwargs: dict[str, str] = {"task": task}
    if language != "auto":
        generate_kwargs["language"] = language

    kwargs: dict[str, Any] = {
        "generate_kwargs": generate_kwargs,
    }
    if chunk_length_s > 0:
        kwargs["chunk_length_s"] = int(chunk_length_s)

    result = pipe({"array": audio_np, "sampling_rate": sample_rate}, **kwargs)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(result).strip()


class MisoTTSWhisperTranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Audio to transcribe. Connect this output to Miso reference_text.",
                    },
                ),
                "model": (
                    whisper_model_choices(),
                    {
                        "default": "whisper-large-v3-turbo (auto-download)",
                        "tooltip": (
                            "Whisper ASR model. Auto-download options are stored under "
                            "ComfyUI/models/audio_encoders/."
                        ),
                    },
                ),
                "dtype": (
                    WHISPER_DTYPE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Whisper precision. auto uses bf16/fp16 on GPU and fp32 on CPU.",
                    },
                ),
                "language": (
                    WHISPER_LANGUAGE_OPTIONS,
                    {
                        "default": "auto",
                        "tooltip": "Optional language hint for Whisper.",
                    },
                ),
                "task": (
                    WHISPER_TASK_OPTIONS,
                    {
                        "default": "transcribe",
                        "tooltip": "transcribe keeps the source language; translate outputs English.",
                    },
                ),
                "chunk_length_s": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Chunk length in seconds for longer audio. 0 disables chunking.",
                    },
                ),
                "download_if_missing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Download selected auto-download Whisper model if missing.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript",)
    FUNCTION = "transcribe"
    CATEGORY = "Miso TTS"
    DESCRIPTION = "Transcribe ComfyUI AUDIO with Whisper and output transcript text for reference_text."

    def transcribe(
        self,
        audio: dict,
        model: str,
        dtype: str,
        language: str,
        task: str,
        chunk_length_s: int,
        download_if_missing: bool,
    ) -> tuple[str]:
        text = transcribe_audio(
            audio=audio,
            model_name=model,
            dtype=dtype,
            language=language,
            task=task,
            chunk_length_s=int(chunk_length_s),
            download_if_missing=bool(download_if_missing),
        )
        logger.info("Whisper transcript: %s", text)
        return (text,)
