"""ComfyUI nodes for Miso TTS 8B.

This wrapper follows the public MisoLabsAI/MisoTTS inference loop while keeping
ComfyUI model-folder, AUDIO, and VRAM-management integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

__version__ = "0.1.0"

logger = logging.getLogger("MisoTTS")
logger.propagate = False

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[MisoTTS] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

try:
    from .loader import register_model_folder
    from .nodes import MisoTTSGenerate, MisoTTSLoadModel
    from .whisper import MisoTTSWhisperTranscribe, register_audio_encoders_folder

    register_model_folder()
    register_audio_encoders_folder()

    NODE_CLASS_MAPPINGS.update(
        {
            "MisoTTSLoadModel": MisoTTSLoadModel,
            "MisoTTSGenerate": MisoTTSGenerate,
            "MisoTTSWhisperTranscribe": MisoTTSWhisperTranscribe,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "MisoTTSLoadModel": "Miso TTS - Load Model",
            "MisoTTSGenerate": "Miso TTS - Generate",
            "MisoTTSWhisperTranscribe": "Miso TTS - Whisper Transcribe",
        }
    )

    logger.info(
        "Registered %d nodes (v%s): %s",
        len(NODE_CLASS_MAPPINGS),
        __version__,
        ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values()),
    )
except Exception as exc:
    logger.error("Failed to register MisoTTS nodes: %s", exc, exc_info=True)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
