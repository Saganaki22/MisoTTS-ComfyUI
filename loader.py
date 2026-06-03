"""Model download, loading, dtype/device, and ComfyUI VRAM integration."""

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .miso_model import (
    HFCSMMisoModel,
    MisoTTSModel,
    TorchtuneMisoTTSModel,
    create_miso_config,
    load_safetensors_into_hf_csm,
    load_safetensors_into_model,
)

logger = logging.getLogger("MisoTTS")

MODEL_REPO_ID = "MisoLabs/MisoTTS"
MODELS_FOLDER_NAME = "misotts"
WEIGHTS_NAME = "model.safetensors"
BF16_WEIGHTS_NAME = "misotts-bf16.safetensors"
DEFAULT_TEXT_TOKENIZER = "meta-llama/Llama-3.2-1B"
DEFAULT_MIMI_CODEC = "kyutai/mimi"

DTYPE_OPTIONS = ["auto", "fp16", "bf16", "fp32"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "sageattention"]
LEGACY_ATTENTION_ALIASES = {"flashattention": "flash_attention"}
ROPE_OPTIONS = ["llama3_scaled", "default"]
_ACTIVE_BUNDLE: MisoTTSBundle | None = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None

MODEL_PRESETS = {
    "MisoTTS 8B Official (fp32)": {
        "repo_id": "MisoLabs/MisoTTS",
        "local_name": WEIGHTS_NAME,
        "filenames": [WEIGHTS_NAME],
    },
    "MisoTTS 8B BF16 (drbaph)": {
        "repo_id": "drbaph/MisoTTS-BF16",
        "local_name": BF16_WEIGHTS_NAME,
        "filenames": [
            BF16_WEIGHTS_NAME,
            WEIGHTS_NAME,
            "MisoTTS-BF16.safetensors",
            "model-bf16.safetensors",
        ],
    },
}
LEGACY_MODEL_ALIASES = {
    WEIGHTS_NAME: "MisoTTS 8B Official (fp32)",
    BF16_WEIGHTS_NAME: "MisoTTS 8B BF16 (drbaph)",
    "model-bf16.safetensors": "MisoTTS 8B BF16 (drbaph)",
}


@dataclass
class MisoTTSBundle:
    model: MisoTTSModel
    patcher: Any
    model_dir: Path
    weights_path: Path
    device: str
    torch_dtype: torch.dtype
    dtype_name: str
    attention: str
    rope: str
    max_seq_len: int
    implementation: str
    download_if_missing: bool
    generator_cache: dict[tuple[str, str], Any] = field(default_factory=dict)


def _get_models_base() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODELS_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parent / "models" / MODELS_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, str(_get_models_base()))
        logger.info("Model folder registered: %s", _get_models_base())
    except Exception:
        pass


def model_dir() -> Path:
    return _get_models_base()


def weights_path() -> Path:
    return model_dir() / WEIGHTS_NAME


def get_model_choices() -> list[str]:
    return list(MODEL_PRESETS)


def _model_preset(model_name: str) -> dict[str, Any] | None:
    return MODEL_PRESETS.get(LEGACY_MODEL_ALIASES.get(model_name, model_name))


def _selected_weights_path(model_name: str) -> Path:
    preset = _model_preset(model_name)
    safe_name = (preset["local_name"] if preset else model_name or WEIGHTS_NAME).replace("\\", "/").lstrip("/")
    path = (model_dir() / safe_name).resolve()
    base = model_dir().resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise ValueError(f"Invalid MisoTTS model path: {model_name}")
    return path


def is_model_downloaded(model_name: str = WEIGHTS_NAME) -> bool:
    path = _selected_weights_path(model_name)
    return path.is_file() and path.stat().st_size > 0


def _download_preset_model(preset: dict[str, Any], dest: Path) -> Path:
    from huggingface_hub import hf_hub_download
    import warnings

    repo_id = preset["repo_id"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for filename in preset["filenames"]:
        try:
            logger.info("Downloading %s/%s to %s", repo_id, filename, dest.parent)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(dest.parent),
                        local_dir_use_symlinks=False,
                    )
            except TypeError:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest.parent),
                )

            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != dest.resolve():
                if dest.exists():
                    dest.unlink()
                shutil.move(str(downloaded_path), str(dest))
            logger.info("MisoTTS model ready at %s", dest)
            return dest
        except Exception as exc:
            last_error = exc

    candidates = ", ".join(preset["filenames"])
    raise FileNotFoundError(
        f"Could not download {preset['repo_id']} to {dest}. Tried: {candidates}. "
        f"Last error: {last_error}"
    )


def download_model_if_missing(model_name: str, download_if_missing: bool) -> Path:
    selected = _selected_weights_path(model_name)
    if is_model_downloaded(model_name):
        return selected
    if not download_if_missing:
        raise FileNotFoundError(
            f"{model_name} is missing. Expected it at {selected}. "
            "Enable download_if_missing or add the file manually."
        )

    preset = _model_preset(model_name)
    if preset is not None:
        return _download_preset_model(preset, selected)

    raise FileNotFoundError(
        f"{model_name} is missing at {selected}. Auto-download is only available for built-in MisoTTS presets."
    )


def resolve_device() -> torch.device:
    try:
        import comfy.model_management as mm

        return torch.device(mm.get_torch_device())
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")


def _dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).removeprefix("torch.")


def _infer_checkpoint_dtype(ckpt: Path) -> torch.dtype | None:
    dtype_map = {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
    }
    try:
        from safetensors import safe_open

        with safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                try:
                    dtype_name = handle.get_slice(key).get_dtype()
                    dtype = dtype_map.get(dtype_name)
                    if dtype is not None:
                        return dtype
                except Exception:
                    dtype = handle.get_tensor(key).dtype
                    if dtype in (torch.float16, torch.bfloat16, torch.float32):
                        return dtype
    except Exception as exc:
        logger.warning("Could not infer dtype from %s: %s", ckpt, exc)
    return None


def _device_default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    if device.type == "xpu":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_dtype(dtype_name: str, device: torch.device, ckpt: Path | None = None) -> torch.dtype:
    if dtype_name == "auto":
        if ckpt is not None:
            checkpoint_dtype = _infer_checkpoint_dtype(ckpt)
            if checkpoint_dtype is not None:
                logger.info(
                    "Auto dtype resolved to %s from checkpoint tensor dtype.",
                    _dtype_to_name(checkpoint_dtype),
                )
                if checkpoint_dtype == torch.bfloat16 and device.type == "cuda":
                    try:
                        if not torch.cuda.is_bf16_supported():
                            logger.warning("Checkpoint is bf16, but this CUDA device does not report bf16 support.")
                    except Exception:
                        pass
                return checkpoint_dtype
        fallback = _device_default_dtype(device)
        logger.info("Auto dtype could not read checkpoint dtype; using %s for %s.", _dtype_to_name(fallback), device)
        return fallback
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        if device.type == "cuda":
            try:
                if not torch.cuda.is_bf16_supported():
                    logger.warning("bf16 requested, but this CUDA device does not report bf16 support.")
            except Exception:
                pass
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _has_flash_attention() -> bool:
    try:
        from flash_attn import flash_attn_func  # noqa: F401

        return True
    except Exception as exc:
        logger.info("FlashAttention not available for auto attention: %s", exc)
        return False


def _has_sageattention() -> bool:
    try:
        from sageattention import sageattn  # noqa: F401

        return True
    except Exception as exc:
        logger.info("SageAttention not available for auto attention: %s", exc)
        return False


def _normalize_attention_name(attention: str) -> str:
    return LEGACY_ATTENTION_ALIASES.get(attention, attention)


def resolve_attention(attention: str, device: torch.device, dtype: torch.dtype) -> str:
    attention = _normalize_attention_name(attention)
    if attention == "auto":
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            if _has_flash_attention():
                logger.info("Auto attention resolved to flash_attention (flash_attn is installed).")
                return "flash_attention"
            if _has_sageattention():
                logger.info("Auto attention resolved to sageattention (sageattention is installed).")
                return "sageattention"
        else:
            logger.info(
                "Auto attention skipping flash_attention/sageattention because device=%s dtype=%s; using sdpa.",
                device,
                _dtype_to_name(dtype),
            )
        logger.info("Auto attention resolved to sdpa.")
        return "sdpa"
    return attention


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _create_model(
    ckpt: Path,
    dtype: torch.dtype,
    device: torch.device,
    attention_backend: str,
    max_seq_len: int,
    rope: str,
) -> tuple[torch.nn.Module, str]:
    if _env_flag("MISO_TTS_USE_TRANSFORMERS_CSM"):
        try:
            logger.info("Trying Transformers native CSM backend with %s RoPE.", rope)
            with torch.device("meta"):
                model = HFCSMMisoModel(
                    max_seq_len=max_seq_len,
                    attention_backend=attention_backend,
                    rope=rope,
                )
            model.to_empty(device=device)
            model.to(dtype=dtype)
            model.device = device
            load_safetensors_into_hf_csm(model, ckpt)
            return model, "transformers_csm"
        except Exception as exc:
            logger.warning(
                "Transformers native CSM backend failed; falling back to reconstructed backend: %s",
                exc,
            )

    if not _env_flag("MISO_TTS_FORCE_RECONSTRUCTED"):
        try:
            if rope != "llama3_scaled":
                raise ValueError("The official torchtune backend only supports Miso's llama3_scaled RoPE.")
            logger.info("Using official MisoLabsAI torchtune CSM backend.")
            config = create_miso_config(max_seq_len=max_seq_len)
            with torch.device("meta"):
                model = TorchtuneMisoTTSModel(config, attention_backend=attention_backend)
            model.to_empty(device=device)
            model.to(dtype=dtype)
            model.device = device
            load_safetensors_into_model(model, ckpt)
            return model, "official_torchtune"
        except Exception as exc:
            logger.warning(
                "Official torchtune backend unavailable; falling back to reconstructed backend. "
                "Install torchtune==0.4.0 into the ComfyUI venv if generated audio is garbled. Error: %s",
                exc,
            )

    logger.info("Using reconstructed Sesame-style CSM backend with %s RoPE.", rope)
    config = create_miso_config(max_seq_len=max_seq_len)
    with torch.device("meta"):
        model = MisoTTSModel(config, attention_backend=attention_backend, dtype=dtype)

    model.to_empty(device=device)
    model.device = device
    model.build_rope_cache(device)
    load_safetensors_into_model(model, ckpt)
    return model, "reconstructed"


def _empty_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()


def _model_has_meta_tensors(model: torch.nn.Module) -> bool:
    return any(tensor.device.type == "meta" for tensor in _module_unique_tensors(model))


def _module_size(model: torch.nn.Module) -> int:
    return sum(tensor.nelement() * tensor.element_size() for tensor in _module_unique_tensors(model))


def _canonical_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    index = device.index
    if index is None and device.type == "cuda" and torch.cuda.is_available():
        try:
            index = torch.cuda.current_device()
        except Exception:
            index = 0
    return torch.device(device.type, index)


def _same_device(left: torch.device, right: torch.device) -> bool:
    left = _canonical_device(left)
    right = _canonical_device(right)
    return left.type == right.type and left.index == right.index


def _module_unique_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for tensor in list(model.parameters(recurse=True)) + list(model.buffers(recurse=True)):
        if tensor is None:
            continue
        key = id(tensor)
        if key in seen:
            continue
        seen.add(key)
        tensors.append(tensor)
    return tensors


class MisoTTSVBar:
    page_size: int = 32 * 1024 * 1024

    def __init__(self, model: MisoTTSModel, device: torch.device):
        self.model = model
        self.device = _canonical_device(device)
        self.tensors: list[torch.Tensor] = []
        self.total_size = 0
        self.total_pages = 1
        self.watermark = 0
        self._refresh_tensors()

    @property
    def offset(self) -> int:
        return self.total_size

    def _refresh_tensors(self) -> None:
        self.tensors = _module_unique_tensors(self.model)
        self.total_size = sum(tensor.nelement() * tensor.element_size() for tensor in self.tensors)
        self.total_pages = max(1, math.ceil(self.total_size / self.page_size)) if self.total_size > 0 else 0

    def loaded_size(self) -> int:
        self._refresh_tensors()
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.tensors
            if _same_device(tensor.device, self.device)
        )

    def get_residency(self) -> list[int]:
        self._refresh_tensors()
        if self.total_size <= 0:
            return []

        residency = [0 for _ in range(self.total_pages)]
        cursor = 0
        for tensor in self.tensors:
            size = tensor.nelement() * tensor.element_size()
            if size <= 0:
                continue
            if _same_device(tensor.device, self.device):
                start_page = cursor // self.page_size
                end_page = min(self.total_pages - 1, (cursor + size - 1) // self.page_size)
                for page in range(start_page, end_page + 1):
                    residency[page] |= 1
            cursor += size
        return residency

    def get_watermark(self) -> int:
        self.watermark = max(self.watermark, self.loaded_size())
        return self.watermark

    def prioritize(self) -> None:
        self.watermark = self.loaded_size()


try:
    import comfy.model_patcher as _model_patcher

    class MisoTTSPatcher(_model_patcher.ModelPatcher):
        def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
            super().__init__(model, load_device, offload_device, size, weight_inplace_update)
            self.hard_unload_to_meta = False
            self._misotts_bundle_ref = None
            self._ensure_dynamic_state(load_device)

        def is_dynamic(self):
            return True

        def _ensure_dynamic_state(self, device):
            if not hasattr(self.model, "dynamic_vbars"):
                self.model.dynamic_vbars = {}
            if not hasattr(self.model, "dynamic_pins"):
                self.model.dynamic_pins = {}
            if device not in self.model.dynamic_pins:
                try:
                    import comfy_aimdo.host_buffer

                    empty_hostbuf = comfy_aimdo.host_buffer.HostBuffer(0, 0, 0)
                except Exception:
                    empty_hostbuf = None
                self.model.dynamic_pins[device] = {
                    "weights": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "patches": (empty_hostbuf, [], [-1], [0], [0], {}),
                    "hostbufs_initialized": False,
                    "failed": False,
                    "active": False,
                }

        def _vbar_get(self):
            vbars = getattr(self.model, "dynamic_vbars", {})
            if vbars:
                return next(iter(vbars.values()))
            return None

        def _set_model_device(self, device):
            try:
                self.model.device = torch.device(device)
            except Exception:
                pass

        def loaded_size(self):
            vbar = self._vbar_get()
            if vbar is not None:
                return vbar.loaded_size()
            return getattr(self.model, "model_loaded_weight_memory", 0)

        def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
            self._ensure_dynamic_state(torch.device(device_to))
            before = self.loaded_size()
            if _model_has_meta_tensors(self.model):
                bundle = self._misotts_bundle_ref() if self._misotts_bundle_ref is not None else None
                if bundle is None:
                    raise RuntimeError("MisoTTS weights were hard-unloaded, but no reload bundle is available.")
                _reload_bundle_weights(bundle, torch.device(device_to))
            else:
                self.model.to(device_to)
            self._set_model_device(device_to)
            self.model.model_loaded_weight_memory = self.model_size()
            return max(0, self.loaded_size() - before)

        def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
            before = self.loaded_size()
            self.detach()
            return before

        def detach(self, unpatch_all=True):
            try:
                if self.hard_unload_to_meta:
                    bundle = self._misotts_bundle_ref() if self._misotts_bundle_ref is not None else None
                    if bundle is not None:
                        for generator in list(bundle.generator_cache.values()):
                            _unload_runtime_codec(generator, unregister=False)
                        bundle.generator_cache.clear()
                    self.model.to_empty(device=torch.device("meta"))
                    self._set_model_device(torch.device("meta"))
                else:
                    self.model.to(self.offload_device)
                    self._set_model_device(self.offload_device)
                self.model.model_loaded_weight_memory = 0
            except Exception:
                pass
            try:
                _empty_accelerator_cache()
            except Exception:
                pass
            return self.model

        def current_loaded_device(self):
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return self.offload_device

        def loaded_ram_size(self):
            return 0

        def pinned_memory_size(self):
            pin_state = getattr(self.model, "dynamic_pins", {}).get(self.load_device)
            if pin_state is None:
                return 0
            return pin_state["weights"][3][0]

        def unregister_inactive_pins(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

        def partially_unload_ram(self, ram_to_unload, subsets=["weights", "patches"]):
            return 0

    del _model_patcher
except Exception:
    MisoTTSPatcher = None


def _register_with_comfy(patcher: Any) -> None:
    if patcher is None or patcher.load_device.type == "cpu":
        return

    try:
        import comfy.model_management as mm
        import weakref

        if any(loaded.model is patcher for loaded in mm.current_loaded_models):
            return

        raw = patcher.model
        if hasattr(patcher, "_ensure_dynamic_state"):
            patcher._ensure_dynamic_state(patcher.load_device)
        raw.model_loaded_weight_memory = patcher.loaded_size()
        raw.dynamic_vbars = {patcher.load_device: MisoTTSVBar(raw, patcher.load_device)}

        loaded = mm.LoadedModel(patcher)
        loaded.real_model = weakref.ref(raw)
        loaded.model_finalizer = weakref.finalize(raw, mm.cleanup_models)
        loaded.model_finalizer.atexit = False
        loaded.currently_used = True

        mm.current_loaded_models.insert(0, loaded)
        logger.info(
            "Registered MisoTTS with ComfyUI VRAM management (%.1f MB).",
            patcher.model_size() / (1024 * 1024),
        )
    except Exception as exc:
        logger.warning("Could not register MisoTTS with ComfyUI VRAM management: %s", exc)


def register_runtime_module(module: torch.nn.Module, device: torch.device) -> Any:
    """Wrap an auxiliary runtime module so Comfy/Aimdo can track its tensors."""
    device = torch.device(device)
    if MisoTTSPatcher is None or device.type == "cpu":
        module.to(device)
        return None

    patcher = MisoTTSPatcher(
        module,
        load_device=device,
        offload_device=torch.device("cpu"),
    )
    module.model_loaded_weight_memory = patcher.model_size()
    _register_with_comfy(patcher)
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    if patcher is None:
        return
    patcher.partially_load(torch.device(device))
    _register_with_comfy(patcher)


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        mm.current_loaded_models[:] = [
            loaded for loaded in mm.current_loaded_models if loaded.model is not patcher
        ]
    except Exception:
        pass


def _unload_runtime_codec(generator: Any, unregister: bool = True) -> None:
    audio_tokenizer = getattr(generator, "audio_tokenizer", None)
    if audio_tokenizer is None:
        return

    patcher = getattr(audio_tokenizer, "patcher", None)
    if patcher is not None:
        if unregister:
            _unregister_from_comfy(patcher)
        try:
            patcher.detach()
        except Exception:
            pass

    codec_model = getattr(audio_tokenizer, "model", None)
    if hasattr(codec_model, "to"):
        try:
            codec_model.to("cpu")
        except Exception:
            pass
    try:
        audio_tokenizer.device = torch.device("cpu")
    except Exception:
        pass


def _reload_bundle_weights(bundle: MisoTTSBundle, device: torch.device) -> None:
    device = torch.device(device)
    logger.info("Reloading hard-unloaded MisoTTS weights from %s to %s.", bundle.weights_path, device)
    model = bundle.model
    if _model_has_meta_tensors(model):
        model.to_empty(device=device)
    else:
        model.to(device)
    model.to(dtype=bundle.torch_dtype)
    model.device = device
    if hasattr(model, "set_attention_backend"):
        model.set_attention_backend(bundle.attention)
    if hasattr(model, "build_rope_cache"):
        model.build_rope_cache(device)
    if bundle.implementation == "transformers_csm":
        load_safetensors_into_hf_csm(model, bundle.weights_path)
    else:
        load_safetensors_into_model(model, bundle.weights_path)
    model.eval()
    model.model_loaded_weight_memory = _module_size(model)


def unload_miso_bundle(bundle: MisoTTSBundle | None, reason: str = "model switch") -> None:
    if bundle is None:
        return

    logger.info("Unloading previous MisoTTS bundle (%s).", reason)
    try:
        for generator in list(bundle.generator_cache.values()):
            _unload_runtime_codec(generator)
        bundle.generator_cache.clear()
    except Exception as exc:
        logger.warning("Could not fully clear MisoTTS generator cache: %s", exc)

    if bundle.patcher is not None:
        _unregister_from_comfy(bundle.patcher)
        try:
            bundle.patcher.detach()
        except Exception as exc:
            logger.warning("Could not detach previous MisoTTS patcher: %s", exc)
    else:
        try:
            bundle.model.to("cpu")
            bundle.model.device = torch.device("cpu")
        except Exception:
            pass

    try:
        bundle.model.model_loaded_weight_memory = 0
        if hasattr(bundle.model, "dynamic_vbars"):
            bundle.model.dynamic_vbars.clear()
        if hasattr(bundle.model, "dynamic_pins"):
            bundle.model.dynamic_pins.clear()
    except Exception:
        pass

    _empty_accelerator_cache()


def load_miso_bundle(
    model_name: str,
    dtype_name: str,
    attention: str,
    download_if_missing: bool,
    max_seq_len: int,
    rope: str = "llama3_scaled",
) -> MisoTTSBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    ckpt = download_model_if_missing(model_name, download_if_missing)
    if download_if_missing:
        from .runtime import ensure_generation_assets

        ensure_generation_assets(
            model_dir(),
            DEFAULT_TEXT_TOKENIZER,
            DEFAULT_MIMI_CODEC,
            download_if_missing=True,
        )
    device = resolve_device()
    dtype = resolve_dtype(dtype_name, device, ckpt)
    attention = _normalize_attention_name(attention)
    using_default_official_backend = not _env_flag("MISO_TTS_USE_TRANSFORMERS_CSM") and not _env_flag("MISO_TTS_FORCE_RECONSTRUCTED")
    if using_default_official_backend and attention == "auto":
        attention_backend = "sdpa"
        logger.info("Auto attention resolved to sdpa for the official torchtune backend.")
    else:
        attention_backend = resolve_attention(attention, device, dtype)

    if attention not in ATTENTION_OPTIONS:
        raise ValueError(f"Unsupported attention backend: {attention}")
    if rope not in ROPE_OPTIONS:
        raise ValueError(f"Unsupported RoPE mode: {rope}")

    load_key = (
        str(ckpt.resolve()),
        ckpt.stat().st_mtime_ns if ckpt.exists() else 0,
        str(dtype),
        attention_backend,
        int(max_seq_len),
        rope,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        logger.info("Reusing active MisoTTS bundle for %s.", ckpt)
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE

    if _ACTIVE_BUNDLE is not None:
        unload_miso_bundle(_ACTIVE_BUNDLE, reason="load settings changed")
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None

    logger.info(
        "Loading MisoTTS from %s on %s with requested dtype=%s resolved dtype=%s, %s attention, and %s RoPE",
        ckpt,
        device,
        dtype_name,
        _dtype_to_name(dtype),
        attention_backend,
        rope,
    )

    model, implementation = _create_model(
        ckpt=ckpt,
        dtype=dtype,
        device=device,
        attention_backend=attention_backend,
        max_seq_len=max_seq_len,
        rope=rope,
    )
    model.eval()
    attention_backend = getattr(model, "attention_backend", attention_backend)

    if MisoTTSPatcher is not None:
        patcher = MisoTTSPatcher(
            model,
            load_device=device,
            offload_device=torch.device("cpu"),
        )
    else:
        patcher = None

    if patcher is not None:
        model.model_loaded_weight_memory = patcher.model_size()
    else:
        model.model_loaded_weight_memory = _module_size(model)

    bundle = MisoTTSBundle(
        model=model,
        patcher=patcher,
        model_dir=model_dir(),
        weights_path=ckpt,
        device=str(device),
        torch_dtype=dtype,
        dtype_name=dtype_name,
        attention=attention_backend,
        rope=rope,
        max_seq_len=max_seq_len,
        implementation=implementation,
        download_if_missing=download_if_missing,
    )
    if patcher is not None:
        patcher.hard_unload_to_meta = True
        patcher._misotts_bundle_ref = weakref.ref(bundle)
        _register_with_comfy(patcher)

    logger.info("MisoTTS loaded successfully using %s backend.", implementation)
    _ACTIVE_BUNDLE = bundle
    _ACTIVE_LOAD_KEY = load_key
    return bundle


def resume_bundle_to_device(bundle: MisoTTSBundle) -> None:
    device = torch.device(bundle.device)
    if bundle.patcher is not None:
        bundle.patcher.partially_load(device)
        _register_with_comfy(bundle.patcher)
    else:
        if _model_has_meta_tensors(bundle.model):
            _reload_bundle_weights(bundle, device)
        else:
            bundle.model.to(device)
            bundle.model.device = device
            bundle.model.model_loaded_weight_memory = _module_size(bundle.model)

    bundle.model.setup_caches(1)
    for generator in bundle.generator_cache.values():
        try:
            generator.to(device)
        except Exception:
            pass
