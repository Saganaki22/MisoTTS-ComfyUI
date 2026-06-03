"""Runtime helpers for text/audio tokenization and CSM-style generation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Optional

import torch

logger = logging.getLogger("MisoTTS")


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x3040 <= cp <= 0x30FF
        or 0x30A0 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0x0E00 <= cp <= 0x0E7F
        or 0x0E80 <= cp <= 0x0EFF
        or 0x1000 <= cp <= 0x109F
        or 0x1780 <= cp <= 0x17FF
    )


def _chunk_by_characters(text: str, chars_per_chunk: int, sentence_end: re.Pattern[str]) -> list[str]:
    if len(text) <= chars_per_chunk:
        return [text.strip()]

    chunks: list[str] = []
    pos = 0
    text_len = len(text)

    while pos < text_len:
        while pos < text_len and text[pos].isspace():
            pos += 1
        if pos >= text_len:
            break

        target_end = min(pos + chars_per_chunk, text_len)
        if target_end >= text_len:
            remaining = text[pos:].strip()
            if remaining:
                chunks.append(remaining)
            break

        segment = text[pos:target_end]
        matches = list(sentence_end.finditer(segment))
        if matches:
            split_at = pos + matches[-1].end()
            chunk = text[pos:split_at].strip()
            if chunk:
                chunks.append(chunk)
            pos = split_at
        else:
            chunk = text[pos:target_end].strip()
            if chunk:
                chunks.append(chunk)
            pos = target_end

    return chunks or [text.strip()]


def _normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentence_units(text: str, sentence_end: re.Pattern[str]) -> list[str]:
    units: list[str] = []
    start = 0
    for match in sentence_end.finditer(text):
        end = match.end()
        unit = _normalize_chunk_text(text[start:end])
        if unit:
            units.append(unit)
        start = end

    trailing = _normalize_chunk_text(text[start:])
    if trailing:
        units.append(trailing)
    return units or [_normalize_chunk_text(text)]


def _split_long_word_unit(text: str, words_per_chunk: int) -> list[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [_normalize_chunk_text(text)]

    chunks: list[str] = []
    current: list[str] = []
    current_count = 0
    clause_end = re.compile(r'[,;:]+["\')\]]?$')

    for word in words:
        current.append(word)
        current_count += 1
        if current_count >= words_per_chunk:
            split_index = None
            for idx in range(len(current) - 1, max(0, len(current) // 2) - 1, -1):
                if clause_end.search(current[idx]):
                    split_index = idx + 1
                    break
            if split_index is None:
                split_index = len(current)
            chunk = " ".join(current[:split_index]).strip()
            if chunk:
                chunks.append(chunk)
            current = current[split_index:]
            current_count = len(current)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def split_text_at_sentence_boundaries(text: str, words_per_chunk: int) -> list[str]:
    """Pack text into chunks at sentence boundaries, falling back only when needed."""
    text = text.strip()
    if not text or words_per_chunk <= 0:
        return [text] if text else []

    sentence_end = re.compile(
        r'(?:[.!?]+["\')\]]?(?:\s|$)|[\u3002\uff1f\uff01\u0964\u0965\u061f\u104b\u0f0d]+)'
    )

    cjk_count = sum(1 for ch in text if _is_cjk(ch))
    alpha_count = sum(1 for ch in text if ch.isalpha() or _is_cjk(ch))
    if alpha_count > 0 and cjk_count / alpha_count > 0.3:
        return _chunk_by_characters(text, words_per_chunk, sentence_end)

    sentence_units = _split_sentence_units(text, sentence_end)
    if len(sentence_units) == 1 and len(sentence_units[0].split()) <= words_per_chunk:
        return sentence_units

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    def flush_current() -> None:
        nonlocal current_sentences, current_word_count
        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())
        current_sentences = []
        current_word_count = 0

    for sentence in sentence_units:
        sentence_word_count = len(sentence.split())
        if sentence_word_count > words_per_chunk:
            flush_current()
            chunks.extend(_split_long_word_unit(sentence, words_per_chunk))
            continue

        if current_sentences and current_word_count + sentence_word_count > words_per_chunk:
            flush_current()

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    flush_current()

    return chunks or [text]


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def _snapshot_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    files: list[str] = []
    for item in path.rglob("*"):
        if item.is_file() and ".cache" not in item.parts:
            files.append(item.relative_to(path).as_posix())
    return sorted(files)


def _describe_snapshot(path: Path) -> str:
    files = _snapshot_files(path)
    if not files:
        return "no files outside .cache"
    preview = ", ".join(files[:12])
    suffix = "" if len(files) <= 12 else f", ... ({len(files)} files)"
    return f"{preview}{suffix}"


def _has_tokenizer_snapshot(path: Path) -> bool:
    has_payload = (
        (path / "tokenizer.json").is_file()
        or (path / "tokenizer.model").is_file()
        or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
        or any(path.glob("*.tiktoken"))
    )
    has_config = (path / "tokenizer_config.json").is_file() or (path / "config.json").is_file()
    return path.exists() and has_payload and has_config


def _has_model_snapshot(path: Path) -> bool:
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return path.exists() and (path / "config.json").is_file() and has_weights


def _ensure_moshi_mimi_snapshot(local_root: Path, download_if_missing: bool) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
    except ImportError:
        return None

    repo_id = loaders.DEFAULT_REPO
    filename = loaders.MIMI_NAME
    dest = local_root / _safe_repo_name(repo_id)
    weight_path = dest / filename
    if weight_path.is_file():
        return weight_path
    if not download_if_missing:
        return None

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Moshi Mimi codec %s/%s to %s", repo_id, filename, dest)
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest))
    return weight_path if weight_path.is_file() else None


def _download_snapshot(
    repo_id: str,
    dest: Path,
    allow_patterns: list[str],
) -> None:
    from huggingface_hub import snapshot_download
    import warnings

    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(dest),
        "allow_patterns": allow_patterns,
        "ignore_patterns": ["*.msgpack", "*.h5", "tf_model*", "flax_model*"],
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)


def _ensure_local_snapshot(
    repo_id_or_path: str,
    root: Path,
    allow_patterns: list[str],
    download_if_missing: bool = True,
    validator: Callable[[Path], bool] | None = None,
    asset_name: str = "asset",
) -> Path:
    source = Path(repo_id_or_path)
    if source.exists():
        if validator is not None and not validator(source):
            raise RuntimeError(
                f"Local {asset_name} at {source} is incomplete ({_describe_snapshot(source)})."
            )
        return source

    root.mkdir(parents=True, exist_ok=True)
    dest = root / _safe_repo_name(repo_id_or_path)

    is_ready = validator(dest) if validator is not None else bool(_snapshot_files(dest))
    if not is_ready:
        if not download_if_missing:
            raise FileNotFoundError(
                f"Missing local {asset_name} for {repo_id_or_path} at {dest} "
                f"({_describe_snapshot(dest)})."
            )
        logger.info("Downloading %s to %s", repo_id_or_path, dest)
        _download_snapshot(repo_id_or_path, dest, allow_patterns)

    is_ready = validator(dest) if validator is not None else bool(_snapshot_files(dest))
    if not is_ready:
        raise RuntimeError(
            f"Downloaded {asset_name} from {repo_id_or_path}, but {dest} is incomplete "
            f"({_describe_snapshot(dest)})."
        )

    return dest


def _find_valid_local_snapshot(
    root: Path,
    repo_ids: list[str],
    validator: Callable[[Path], bool],
) -> Path | None:
    for repo_id in repo_ids:
        path = root / _safe_repo_name(repo_id)
        if validator(path):
            return path
    return None


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)


def comfy_audio_to_tensor(audio: dict, target_sr: int) -> torch.Tensor:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)

    wav = waveform[0].float()
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    wav = wav.detach().cpu()

    if sample_rate != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sample_rate, target_sr)

    return wav.contiguous()


def tensor_audio_to_comfy(audio: torch.Tensor, sample_rate: int) -> dict:
    audio = audio.detach().float().cpu()
    if audio.ndim > 1:
        audio = audio.squeeze()
    return {
        "waveform": audio.view(1, 1, -1).contiguous(),
        "sample_rate": int(sample_rate),
    }


class TransformersMimiCodec:
    def __init__(
        self,
        repo_id: str,
        local_root: Path,
        device: torch.device,
        dtype: torch.dtype,
        num_codebooks: int = 32,
        download_if_missing: bool = True,
    ):
        from transformers import MimiModel

        self.repo_id = repo_id
        self.device = device
        self.num_codebooks = num_codebooks
        load_dtype = dtype if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) else torch.float32
        local_path = _ensure_local_snapshot(
            repo_id,
            local_root,
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
            download_if_missing=download_if_missing,
            validator=_has_model_snapshot,
            asset_name="Mimi codec",
        )

        logger.info("Loading Mimi codec from %s", local_path)
        self.model = MimiModel.from_pretrained(
            str(local_path),
            torch_dtype=load_dtype,
            local_files_only=True,
        )
        self.model.to(device)
        self.model.eval()
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", 24000))
        self.codebook_size = int(getattr(self.model.config, "codebook_size", 2048))
        self.patcher = None
        try:
            from .loader import register_runtime_module

            self.patcher = register_runtime_module(self.model, device)
        except Exception as exc:
            logger.warning("Could not register Mimi codec with ComfyUI model management: %s", exc)

    def to(self, device: torch.device) -> None:
        self.device = device
        if self.patcher is not None:
            try:
                from .loader import resume_runtime_module

                resume_runtime_module(self.patcher, device)
                return
            except Exception as exc:
                logger.warning("Could not resume Mimi codec through ComfyUI model management: %s", exc)
        self.model.to(device)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        audio = audio.to(self.device, dtype=model_dtype).view(1, 1, -1)
        encoded = self.model.encode(
            input_values=audio,
            num_quantizers=self.num_codebooks,
            return_dict=True,
        )
        return encoded.audio_codes[0].long()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # Transformers Mimi expects (batch, codebooks, frames).
        codes = codes.to(self.device).long()
        invalid_mask = (codes < 0) | (codes >= self.codebook_size)
        if torch.any(invalid_mask):
            first_bad = codes[invalid_mask][0].item()
            raise RuntimeError(
                f"MisoTTS generated Mimi code {first_bad}, but the codec only accepts "
                f"0..{self.codebook_size - 1}. This usually means the CSM inference path "
                "is misaligned with the checkpoint."
            )
        decoded = self.model.decode(audio_codes=codes, return_dict=True)
        audio = decoded.audio_values
        return audio.squeeze(0).squeeze(0)


class MoshiMimiCodec:
    def __init__(
        self,
        local_root: Path,
        device: torch.device,
        num_codebooks: int = 32,
        download_if_missing: bool = True,
    ):
        from moshi.models import loaders

        self.device = device
        self.num_codebooks = num_codebooks
        self.codebook_size = 2048
        weight_path = _ensure_moshi_mimi_snapshot(local_root, download_if_missing)
        if weight_path is None or not weight_path.is_file():
            raise FileNotFoundError(
                f"Missing Moshi Mimi codec {loaders.MIMI_NAME}. Enable download_if_missing."
            )

        logger.info("Loading Moshi Mimi codec from %s", weight_path)
        self.model = loaders.get_mimi(str(weight_path), device=device)
        self.model.set_num_codebooks(num_codebooks)
        self.sample_rate = int(self.model.sample_rate)
        self.patcher = None
        if isinstance(self.model, torch.nn.Module):
            try:
                from .loader import register_runtime_module

                self.patcher = register_runtime_module(self.model, device)
            except Exception as exc:
                logger.warning("Could not register Moshi Mimi codec with ComfyUI model management: %s", exc)

    def to(self, device: torch.device) -> None:
        self.device = device
        if self.patcher is not None:
            try:
                from .loader import resume_runtime_module

                resume_runtime_module(self.patcher, device)
                return
            except Exception as exc:
                logger.warning("Could not resume Moshi Mimi codec through ComfyUI model management: %s", exc)
        if hasattr(self.model, "to"):
            self.model.to(device)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device).view(1, 1, -1)
        return self.model.encode(audio)[0].long()

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(self.device).long()
        invalid_mask = (codes < 0) | (codes >= self.codebook_size)
        if torch.any(invalid_mask):
            first_bad = codes[invalid_mask][0].item()
            raise RuntimeError(
                f"MisoTTS generated Mimi code {first_bad}, but the codec only accepts "
                f"0..{self.codebook_size - 1}."
            )
        return self.model.decode(codes).squeeze(0).squeeze(0)


def load_mimi_codec(
    repo_id: str,
    local_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    num_codebooks: int,
    download_if_missing: bool,
):
    try:
        return MoshiMimiCodec(
            local_root=local_root,
            device=device,
            num_codebooks=num_codebooks,
            download_if_missing=download_if_missing,
        )
    except ImportError as exc:
        logger.warning("Moshi is not installed; falling back to Transformers Mimi codec: %s", exc)
    except Exception as exc:
        logger.warning("Could not load Moshi Mimi codec; falling back to Transformers Mimi codec: %s", exc)

    return TransformersMimiCodec(
        repo_id,
        local_root,
        device,
        dtype,
        num_codebooks,
        download_if_missing=download_if_missing,
    )


def _tokenizer_patterns() -> list[str]:
    return [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "*.tiktoken",
    ]


def ensure_generation_assets(
    model_dir: Path,
    tokenizer_repo: str,
    mimi_repo: str,
    download_if_missing: bool = True,
) -> tuple[Path, Path]:
    tokenizer_repos = []
    for repo in (tokenizer_repo, "meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"):
        if repo and repo not in tokenizer_repos:
            tokenizer_repos.append(repo)

    tokenizer_path = model_dir if _has_tokenizer_snapshot(model_dir) else None
    if tokenizer_path is None:
        tokenizer_path = _find_valid_local_snapshot(
            model_dir / "tokenizers",
            tokenizer_repos,
            _has_tokenizer_snapshot,
        )

    last_error: Exception | None = None
    if tokenizer_path is None:
        for repo in tokenizer_repos:
            try:
                tokenizer_path = _ensure_local_snapshot(
                    repo,
                    model_dir / "tokenizers",
                    _tokenizer_patterns(),
                    download_if_missing=download_if_missing,
                    validator=_has_tokenizer_snapshot,
                    asset_name="text tokenizer",
                )
                break
            except Exception as exc:
                last_error = exc
                logger.warning("Could not prepare tokenizer asset %s: %s", repo, exc)

    if tokenizer_path is None:
        raise RuntimeError(f"Failed to prepare tokenizer assets. Last error: {last_error}")

    moshi_path = _ensure_moshi_mimi_snapshot(model_dir / "codecs", download_if_missing)
    if moshi_path is not None and moshi_path.is_file():
        return tokenizer_path, moshi_path

    mimi_path = _ensure_local_snapshot(
        mimi_repo,
        model_dir / "codecs",
        ["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
        download_if_missing=download_if_missing,
        validator=_has_model_snapshot,
        asset_name="Mimi codec",
    )
    return tokenizer_path, moshi_path or mimi_path


def load_llama_tokenizer(
    tokenizer_repo: str,
    local_model_dir: Path | None = None,
    download_if_missing: bool = True,
):
    from tokenizers.processors import TemplateProcessing
    from transformers import AutoTokenizer

    candidates: list[tuple[str, Path | str, Path | None]] = []
    if local_model_dir is not None:
        tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        if any((local_model_dir / name).exists() for name in tokenizer_files):
            candidates.append(("path", local_model_dir, None))

    tokenizer_repos: list[str] = []
    if tokenizer_repo:
        tokenizer_repos.append(tokenizer_repo)
    if local_model_dir is not None:
        for repo in ("meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"):
            if repo not in tokenizer_repos:
                tokenizer_repos.append(repo)

        tokenizers_root = local_model_dir / "tokenizers"
        for repo in tokenizer_repos:
            path = tokenizers_root / _safe_repo_name(repo)
            if _has_tokenizer_snapshot(path):
                candidates.append(("path", path, None))

    for repo in tokenizer_repos:
        candidates.append(("repo", repo, (local_model_dir or Path.cwd()) / "tokenizers"))

    last_error: Exception | None = None
    tokenizer_patterns = _tokenizer_patterns()

    for kind, candidate, root in candidates:
        try:
            source = (
                _ensure_local_snapshot(
                    str(candidate),
                    root,
                    tokenizer_patterns,
                    download_if_missing=download_if_missing,
                    validator=_has_tokenizer_snapshot,
                    asset_name="text tokenizer",
                )
                if kind == "repo" and root is not None
                else Path(candidate)
            )
            if not _has_tokenizer_snapshot(Path(source)):
                raise RuntimeError(
                    f"Tokenizer path {source} is incomplete ({_describe_snapshot(Path(source))})."
                )
            tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
            bos = tokenizer.bos_token
            eos = tokenizer.eos_token
            if bos is not None and eos is not None and hasattr(tokenizer, "_tokenizer"):
                tokenizer._tokenizer.post_processor = TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                    special_tokens=[
                        (bos, tokenizer.bos_token_id),
                        (eos, tokenizer.eos_token_id),
                    ],
                )
            logger.info("Loaded text tokenizer from %s", source)
            return tokenizer
        except Exception as exc:
            last_error = exc
            logger.warning("Could not load tokenizer from %s: %s", candidate, exc)

    raise RuntimeError(f"Failed to load a Llama 3.2 tokenizer. Last error: {last_error}")


class MisoTTSGenerator:
    def __init__(
        self,
        bundle,
        tokenizer_repo: str,
        mimi_repo: str,
    ):
        self.bundle = bundle
        self.model = bundle.model
        self.device = torch.device(bundle.device)
        self.text_tokenizer = load_llama_tokenizer(
            tokenizer_repo,
            bundle.model_dir,
            download_if_missing=bundle.download_if_missing,
        )
        self.audio_tokenizer = load_mimi_codec(
            repo_id=mimi_repo,
            local_root=bundle.model_dir / "codecs",
            device=self.device,
            dtype=bundle.torch_dtype,
            num_codebooks=self.model.config.audio_num_codebooks,
            download_if_missing=bundle.download_if_missing,
        )
        self.sample_rate = self.audio_tokenizer.sample_rate
        self.model.setup_caches(1)

    def to(self, device: torch.device) -> None:
        self.device = device
        self.model = self.bundle.model
        self.audio_tokenizer.to(device)

    def _format_text(self, text: str, speaker: int) -> str:
        return f"[{speaker}] {text.lstrip()}"

    def _tokenize_text_segment(
        self,
        text: str,
        speaker: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self.text_tokenizer.encode(self._format_text(text, speaker))
        frame = torch.zeros(len(text_tokens), self.model.config.audio_num_codebooks + 1, dtype=torch.long)
        mask = torch.zeros_like(frame, dtype=torch.bool)
        frame[:, -1] = torch.tensor(text_tokens, dtype=torch.long)
        mask[:, -1] = True
        return frame.to(self.device), mask.to(self.device)

    def _tokenize_audio(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio_tokens = self.audio_tokenizer.encode(audio)
        eos_frame = torch.zeros(audio_tokens.size(0), 1, dtype=audio_tokens.dtype, device=audio_tokens.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        frame = torch.zeros(audio_tokens.size(1), self.model.config.audio_num_codebooks + 1, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(frame, dtype=torch.bool)
        frame[:, :-1] = audio_tokens.transpose(0, 1)
        mask[:, :-1] = True
        return frame, mask

    def _tokenize_segment(
        self,
        text: str,
        speaker: int,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_mask = self._tokenize_text_segment(text, speaker)
        audio_tokens, audio_mask = self._tokenize_audio(audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_mask, audio_mask], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        max_audio_length_ms: int,
        temperature: float,
        top_k: int,
        ref_audio: Optional[torch.Tensor] = None,
        ref_text: str = "",
        ref_speaker: int = 0,
        context_segments: Optional[list[tuple[int, str, torch.Tensor]]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        self.model.reset_caches()
        max_generation_len = max(1, int(max_audio_length_ms / 80))
        max_seq_len = self.model.config.max_seq_len

        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        if context_segments is not None:
            for ctx_speaker, ctx_text, ctx_audio in context_segments:
                if not ctx_text.strip():
                    raise ValueError("Every audio context segment needs matching transcript text.")
                segment_tokens, segment_mask = self._tokenize_segment(
                    ctx_text.strip(),
                    int(ctx_speaker),
                    ctx_audio,
                )
                tokens.append(segment_tokens)
                masks.append(segment_mask)
        elif ref_audio is not None:
            if not ref_text.strip():
                raise ValueError(
                    "reference_text is required when reference_audio is connected. "
                    "Connect Miso TTS - Whisper Transcribe to reference_text or type the transcript."
                )
            segment_tokens, segment_mask = self._tokenize_segment(
                ref_text.strip(),
                ref_speaker,
                ref_audio,
            )
            tokens.append(segment_tokens)
            masks.append(segment_mask)

        gen_tokens, gen_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_tokens)
        masks.append(gen_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_mask = torch.cat(masks, dim=0).bool().to(self.device)

        max_context_len = max_seq_len - max_generation_len
        if prompt_tokens.size(0) >= max_context_len:
            raise ValueError(
                f"Inputs too long for requested generation. Prompt has {prompt_tokens.size(0)} frames, "
                f"but must be below {max_context_len}. Reduce reference audio/text or the duration cap."
            )
        logger.info(
            "Prompt frames=%d, max_generation_frames=%d.",
            prompt_tokens.size(0),
            max_generation_len,
        )

        samples: list[torch.Tensor] = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_mask = prompt_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=self.device).unsqueeze(0).long()
        stopped_by_eos = False

        for idx in range(max_generation_len):
            sample = self.model.generate_frame(
                curr_tokens,
                curr_mask,
                curr_pos,
                temperature=temperature,
                top_k=top_k,
            )
            if torch.all(sample == 0):
                stopped_by_eos = True
                break

            samples.append(sample)
            curr_tokens = torch.cat(
                [
                    sample,
                    torch.zeros(1, 1, dtype=torch.long, device=self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_mask = torch.cat(
                [
                    torch.ones_like(sample, dtype=torch.bool),
                    torch.zeros(1, 1, dtype=torch.bool, device=self.device),
                ],
                dim=1,
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

            if progress is not None:
                progress(idx + 1, max_generation_len)

        if not samples:
            raise RuntimeError("MisoTTS generated no audio frames.")

        logger.info(
            "Generated %d Mimi frames (%.2f seconds), stop=%s.",
            len(samples),
            len(samples) * 0.08,
            "eos" if stopped_by_eos else "duration_cap",
        )
        codes = torch.stack(samples).permute(1, 2, 0)
        all_zero_frames = int(codes.eq(0).all(dim=1).sum().item())
        invalid_codes = int((codes >= self.audio_tokenizer.codebook_size).sum().item())
        logger.info(
            "Mimi code stats: min=%d max=%d invalid_for_codec=%d all_zero_frames=%d/%d.",
            int(codes.min().item()),
            int(codes.max().item()),
            invalid_codes,
            all_zero_frames,
            codes.size(-1),
        )
        if not stopped_by_eos:
            logger.warning(
                "MisoTTS reached the duration cap without EOS; generated audio may be unreliable."
            )
        return self.audio_tokenizer.decode(codes)
