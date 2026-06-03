# MisoTTS-ComfyUI

**Miso TTS 8B nodes for ComfyUI** - Sesame-style CSM text-to-speech with Mimi audio tokens, optional reference-audio continuation, Whisper transcription, ComfyUI AUDIO wiring, and Aimdo/VRAM-management integration.

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/Saganaki22/MisoTTS-ComfyUI)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![MisoTTS Model](https://img.shields.io/badge/%F0%9F%A4%97%20MisoTTS-MisoLabs%2FMisoTTS-blue)](https://huggingface.co/MisoLabs/MisoTTS)
[![MisoTTS BF16](https://img.shields.io/badge/%F0%9F%A4%97%20MisoTTS--BF16-drbaph%2FMisoTTS--BF16-blue)](https://huggingface.co/drbaph/MisoTTS-BF16)
[![Official Code](https://img.shields.io/badge/GitHub%20Official-MisoLabsAI%2FMisoTTS-black)](https://github.com/MisoLabsAI/MisoTTS)
[![Mimi Codec](https://img.shields.io/badge/%F0%9F%A4%97%20Mimi-kyutai%2Fmimi-yellow)](https://huggingface.co/kyutai/mimi)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> <u>Important: reference audio is prompt/context conditioning, not guaranteed speaker-identity cloning.</u> The upstream README describes conditioning on prior audio for voice cloning, but the public inference code implements this as conversational `Segment(text, speaker, audio)` context. It can follow a reference voice sometimes, but it can also drift or change speaker characteristics.

## Features

- **Miso TTS 8B** - Loads the public MisoLabs/MisoTTS safetensor with the official torchtune CSM architecture.
- **BF16 preset** - Optional `drbaph/MisoTTS-BF16` preset for lower VRAM once the BF16 file is available or placed locally.
- **Reference audio conditioning** - Optional prompt audio plus transcript context using the same segment format as the official Miso inference code.
- **Whisper transcription node** - ComfyUI `AUDIO` in, transcript `STRING` out, ready to connect to `reference_text`.
- **Native ComfyUI AUDIO** - Generate node outputs ComfyUI `AUDIO`; use ComfyUI's built-in save nodes.
- **Local asset storage** - Models, tokenizers, and codecs are stored as normal files under `ComfyUI/models/`, not only hidden HF cache blobs.
- **Local-first loading** - If tokenizer/codec files already exist, generation does not check Hugging Face every prompt.
- **Automatic dtype** - `auto` reads the safetensor dtype when possible; fp32 files load fp32, bf16 files load bf16.
- **Aimdo/VRAM visibility** - Main model, Mimi codec, and Whisper model are registered with ComfyUI model management.
- **Hard unload** - Manual ComfyUI unload releases the main Miso weights to meta tensors instead of keeping a giant CPU RAM copy.

## Installation

### Method 1: ComfyUI Manager

Search for **Miso TTS** or **MisoTTS-ComfyUI** in ComfyUI Manager and click Install.

After installation, restart ComfyUI. If the Manager does not install optional Moshi support, install it manually into the same ComfyUI Python environment:

```bash
python install.py
```

### Method 2: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/MisoTTS-ComfyUI.git
cd MisoTTS-ComfyUI
python install.py
```

For ComfyUI portable on Windows, run the install commands with the bundled Python:

```bat
..\..\python_embeded\python.exe install.py
```

For a venv install on Windows:

```bat
..\..\venv\Scripts\python.exe install.py
```

The `requirements.txt` file is a commented dependency reference only. It is intentionally inert so automated installers do not bypass `install.py` and resolve dependency chains on their own.

If you need to install one package manually, always pass `--no-deps`, for example:

```bash
python -m pip install torchtune==0.4.0 --no-deps
python -m uv pip install moshi==0.2.2 --no-deps
```

### Why install Miso runtime packages with `--no-deps`?

The official Miso inference stack uses torchtune plus Kyutai/Moshi's Mimi codec. Installing these with dependency resolution can try to alter packages already managed by ComfyUI. `--no-deps` keeps ComfyUI's Torch stack intact and installs only the requested runtime packages.

## Nodes

<details>
<summary><strong>1. Miso TTS - Load Model</strong> - Load Miso weights with dtype and attention controls</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | COMBO | `MisoTTS 8B Official (fp32)` | Built-in checkpoint preset. |
| dtype | COMBO | `auto` | `auto`, `fp16`, `bf16`, `fp32`. `auto` reads the safetensor dtype when possible. |
| attention | COMBO | `auto` | `auto`, `sdpa`, `flash_attention`, `sageattention`. Official torchtune backend uses mask-aware SDPA for correctness. |
| download_if_missing | BOOLEAN | `True` | Downloads the selected preset plus tokenizer and codec into `ComfyUI/models/misotts/` when missing. |

**Output:**
- `miso_model` - `MISO_MODEL`, connect to Generate.

</details>

<details>
<summary><strong>2. Miso TTS - Generate</strong> - Text to AUDIO with optional reference context</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| miso_model | MISO_MODEL | required | Output from Load Model. |
| text | STRING, multiline | `"Hello!..."` | Text to synthesize. |
| reference_text | STRING, multiline | `""` | Transcript for `reference_audio`. Required when reference audio is connected. |
| speaker | INT | `0` | Prompt speaker tag used as `[{speaker}]`. Leave `0` for normal single-speaker use. |
| max_audio_length_seconds | FLOAT | `10.0` | Maximum generated audio per chunk in seconds. |
| longform_chunking | BOOLEAN | `False` | Splits long text at sentence boundaries. Each chunk reuses the original reference audio/text if connected. |
| words_per_chunk | INT | `80` | Target words per longform chunk. `0` disables text splitting. |
| temperature | FLOAT | `0.9` | Sampling temperature. `0` is greedy. |
| top_k | INT | `50` | Top-k sampling over the 2051 audio vocabulary. |
| seed | INT | `0` | Random seed. `0` chooses a random seed. |

**Optional Input:**
- `reference_audio` - ComfyUI `AUDIO` prompt/reference audio.

**Output:**
- `audio` - ComfyUI `AUDIO`.

</details>

<details>
<summary><strong>3. Miso TTS - Whisper Transcribe</strong> - AUDIO to transcript text</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | AUDIO | required | Audio to transcribe. |
| model | COMBO | `whisper-large-v3-turbo (auto-download)` | Whisper model selection. Auto-download choices are stored under `ComfyUI/models/audio_encoders/`. |
| dtype | COMBO | `auto` | `auto`, `fp16`, `bf16`, `fp32`. |
| language | COMBO | `auto` | Optional language hint. |
| task | COMBO | `transcribe` | `transcribe` keeps source language; `translate` outputs English. |
| chunk_length_s | INT | `30` | Whisper chunk length in seconds. `0` disables chunking. |
| download_if_missing | BOOLEAN | `True` | Download selected Whisper model if missing. |

**Output:**
- `transcript` - `STRING`, connect to `reference_text` on Generate.

</details>

## Generation Parameters Guide

Miso TTS is autoregressive over Mimi frames. It is not diffusion-based, so there is no `steps` quality setting.

| Parameter | What it does | Tips |
|-----------|--------------|------|
| `max_audio_length_seconds` | Upper duration cap per generated chunk | The model may stop earlier if it emits EOS. |
| `temperature` | Sampling randomness | Lower is more stable; `0.9` matches the official default. |
| `top_k` | Restricts token sampling to the top K candidates | `50` matches the official default. |
| `seed` | Controls sampling repeatability | Longform chunks reuse the same seed to reduce voice drift. |
| `speaker` | Text prompt speaker tag | Not a voice preset. Leave at `0` unless building multi-speaker context. |
| `reference_audio` | Audio context for continuation | Best with clean speech and matching `reference_text`. |
| `longform_chunking` | Splits long text at sentence boundaries | Helps texts that exceed the 2048-frame context limit. |
| `words_per_chunk` | Target chunk size for longform | Automatically lowered when needed so chunks can fit inside `max_audio_length_seconds`. |

## Attention Backends

The dropdown exposes multiple labels, but the safe path depends on backend.

| Option | Official torchtune backend | Reconstructed fallback backend |
|--------|----------------------------|--------------------------------|
| `auto` | Resolves to `sdpa` | Tries `flash_attention`, then `sageattention`, then `sdpa` |
| `sdpa` | PyTorch scaled-dot-product attention | PyTorch scaled-dot-product attention |
| `flash_attention` | Coerced to `sdpa` for mask correctness | Uses `flash_attn` when possible, falls back if needed |
| `sageattention` | Coerced to `sdpa` for mask correctness | Uses `sageattention` when possible, falls back if needed |

The official torchtune path uses KV-cache masks during inference. Unmasked flash/sage kernels can produce corrupted audio if they ignore those masks, so the node intentionally uses mask-aware SDPA for the official backend.

## Model Storage

All paths are resolved from `folder_paths.models_dir`, so they work in portable and normal ComfyUI installs on Windows, Linux, and macOS.

```text
ComfyUI/models/
  misotts/
    model.safetensors                       # MisoLabs/MisoTTS fp32
    misotts-bf16.safetensors                # drbaph/MisoTTS-BF16 bf16
    tokenizers/
      meta-llama_Llama-3.2-1B/
      unsloth_Llama-3.2-1B/
    codecs/
      kyutai_moshiko-pytorch-bf16/
        tokenizer-e351c8d8-checkpoint125.safetensors
      kyutai_mimi/                          # Transformers fallback codec
  audio_encoders/
    openai_whisper-large-v3-turbo/
    openai_whisper-large-v3/
    openai_whisper-medium/
    openai_whisper-small/
    openai_whisper-tiny/
```

Hugging Face may create `.cache` metadata folders inside these local snapshots. The usable model/tokenizer/codec payload files are stored directly under `ComfyUI/models/`.

### Available Miso Models

| Dropdown name | Source | Local filename | Notes |
|---------------|--------|----------------|-------|
| `MisoTTS 8B Official (fp32)` | [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) | `model.safetensors` | Original fp32 weights. |
| `MisoTTS 8B BF16 (drbaph)` | [drbaph/MisoTTS-BF16](https://huggingface.co/drbaph/MisoTTS-BF16) | `misotts-bf16.safetensors` | Lower VRAM BF16 weights. |

### Tokenizer and Codec Assets

| Asset | Preferred source | Fallback | Local path |
|-------|------------------|----------|------------|
| Text tokenizer | `meta-llama/Llama-3.2-1B` | `unsloth/Llama-3.2-1B` | `ComfyUI/models/misotts/tokenizers/` |
| Mimi codec | Moshi `kyutai/moshiko-pytorch-bf16` | `kyutai/mimi` through Transformers | `ComfyUI/models/misotts/codecs/` |
| Whisper ASR | `openai/whisper-*` | Manual model folder | `ComfyUI/models/audio_encoders/` |

The loader checks valid local tokenizer and codec folders before any online request.

## VRAM and RAM Notes

| Precision | Approx weight size | Practical VRAM | Notes |
|-----------|--------------------|----------------|-------|
| fp32 | ~32 GB | Much higher than bf16 | Original checkpoint. Heavy VRAM/RAM requirement. |
| bf16 | ~16 GB | ~18-20 GB | Recommended for modern CUDA GPUs with bf16 support. |
| fp16 | ~16 GB | Similar class to bf16 | Can be selected at load time, but may be less faithful than loading a bf16 checkpoint as bf16. |

The **BF16 model** uses around **18-20 GB VRAM** during generation once runtime buffers, KV cache, and codec overhead are included. This number is not for the fp32 model, which is substantially heavier. The nodepack has **Aimdo/ComfyUI model-management integration**, so the main model, Mimi codec, and Whisper model can appear in VRAM visualization and participate in ComfyUI unload behavior.

Manual ComfyUI unload hard-unloads the main transformer to meta tensors, so the 8B model should not remain as a large CPU RAM copy. Reusing a cached node output later reloads weights from the local safetensor.

## Model Caching

Changing the selected model preset, dtype, or attention backend unloads the previous active Miso bundle and clears CUDA/XPU cache. If the selected checkpoint file changes on disk, the loader treats it as a new model load.

The Mimi codec is created when Generate first runs. Whisper is created when the Whisper Transcribe node first runs. Both are registered with ComfyUI model management so Aimdo/VRAM visualization can see their tensors.

## Reference Audio Limits

Miso's public inference code exposes reference audio as conversational context segments:

```text
Segment(speaker=<id>, text=<transcript>, audio=<24kHz mono waveform>)
```

This node mirrors that behavior. It is closer to **voice continuation from prompt audio** than a separate speaker-embedding cloning system. For best results:

<u>Do not assume reference audio guarantees exact voice identity, gender, accent, or speaker matching.</u> The current upstream README says Miso can condition on prior audio for voice cloning, but the public implementation routes reference audio through context segments rather than a dedicated speaker embedding or verifier. Treat it as reference conditioning that may drift.

- Use clean mono speech.
- Provide accurate `reference_text`.
- Keep the reference concise.
- Use the same `speaker` value for reference and generated text unless deliberately building multi-speaker context.
- Connect `Miso TTS - Whisper Transcribe` to `reference_text` if you do not want to type the transcript.

## Troubleshooting

<details>
<summary>Quick fixes for common issues</summary>

### Generated audio is garbled

Use the official torchtune backend with `attention=auto` or `attention=sdpa`. The official backend intentionally uses SDPA because KV-cache masks are required for correct generation.

Restart ComfyUI after updating the nodepack so Python reloads the patched files.

### Tokenizer tries to download every run

Make sure a valid tokenizer folder exists under:

```text
ComfyUI/models/misotts/tokenizers/unsloth_Llama-3.2-1B/
```

The node now uses valid local tokenizer folders first. If it still checks online, the local folder is incomplete.

### Model download fails

Set a Hugging Face mirror before starting ComfyUI if needed:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

Windows batch example:

```bat
set HF_ENDPOINT=https://hf-mirror.com
python main.py
```

### BF16 preset download fails

Check that the BF16 safetensor exists in [drbaph/MisoTTS-BF16](https://huggingface.co/drbaph/MisoTTS-BF16), or place the file manually at:

```text
ComfyUI/models/misotts/misotts-bf16.safetensors
```

### CUDA out of memory

- Use `MisoTTS 8B BF16 (drbaph)`.
- Use `dtype=auto` so bf16 checkpoints stay bf16.
- Reduce `max_audio_length_seconds`.
- Restart ComfyUI after large model unloads if another extension still holds memory.

### Whisper model is not visible in VRAM tools

Run the Whisper Transcribe node once. Whisper is registered with ComfyUI model management only when it first loads.


</details>

## Credits

- **Miso TTS 8B model** - [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) by Miso Labs.
- **Official Miso inference code** - [MisoLabsAI/MisoTTS](https://github.com/MisoLabsAI/MisoTTS).
- **Sesame CSM architecture** - [SesameAILabs/csm](https://github.com/SesameAILabs/csm).
- **Mimi codec** - [kyutai/mimi](https://huggingface.co/kyutai/mimi) and Moshi codec utilities.
- **Whisper ASR** - [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and related OpenAI Whisper checkpoints.
- **ComfyUI nodepack** - [Saganaki22/MisoTTS-ComfyUI](https://github.com/Saganaki22/MisoTTS-ComfyUI).

## Citation

If you use this nodepack, cite the original model/code projects according to their upstream guidance:

- [MisoLabs/MisoTTS model card](https://huggingface.co/MisoLabs/MisoTTS)
- [MisoLabsAI/MisoTTS official code](https://github.com/MisoLabsAI/MisoTTS)
- [SesameAILabs/csm](https://github.com/SesameAILabs/csm)
- [Kyutai Mimi / Moshi paper and model card](https://huggingface.co/kyutai/mimi)

Kyutai Mimi/Moshi citation from the Mimi model card:

```bibtex
@techreport{kyutai2024moshi,
    author = {Alexandre D\\'efossez and Laurent Mazar\\'e and Manu Orsini and Am\\'elie Royer and Patrick P\\'erez and Herv\\'e J\\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year = {2024},
    month = {September},
    url = {http://kyutai.org/Moshi.pdf}
}
```

## License

This custom node is released under the Apache 2.0 License. Miso TTS, Mimi, Moshi, Whisper, torchtune, and related upstream models/code each have their own licenses. Check the linked upstream repositories and model cards before redistribution or commercial use.

## Star History

<a href="https://www.star-history.com/?repos=Saganaki22%2FMisoTTS-ComfyUI&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Saganaki22/MisoTTS-ComfyUI&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Saganaki22/MisoTTS-ComfyUI&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Saganaki22/MisoTTS-ComfyUI&type=date&legend=top-left" />
 </picture>
</a>
