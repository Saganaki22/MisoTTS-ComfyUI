"""Safe installer for MisoTTS-ComfyUI.

ComfyUI owns the Torch stack. This installer avoids dependency resolution for
packages that can otherwise pull or change torch/torchaudio builds.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys


PREFIX = "[MisoTTS]"


def _run(cmd: list[str], timeout: int = 300) -> bool:
    print(f"{PREFIX} Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        print(f"{PREFIX} Install command failed: {exc}")
        return False

    if result.returncode == 0:
        return True

    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    return False


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _version_at_least(package_name: str, minimum: str) -> bool:
    try:
        installed = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return _version_tuple(installed) >= _version_tuple(minimum)


def _pip_install(requirement: str, *, no_deps: bool = True) -> bool:
    flags = ["--no-deps"] if no_deps else []
    uv_cmd = [sys.executable, "-m", "uv", "pip", "install", requirement] + flags
    if _run(uv_cmd):
        return True

    pip_cmd = [sys.executable, "-m", "pip", "install", requirement] + flags
    return _run(pip_cmd)


def _ensure(requirement: str, module_name: str, *, minimum: str | None = None) -> None:
    installed = _has_module(module_name)
    if installed and minimum is not None:
        installed = _version_at_least(module_name, minimum)

    if installed:
        print(f"{PREFIX} {requirement} already available.")
        return

    print(f"{PREFIX} Installing {requirement} with --no-deps.")
    if not _pip_install(requirement, no_deps=True):
        print(f"{PREFIX} WARNING: Failed to install {requirement}. Try manually:")
        print(f"  {sys.executable} -m pip install {requirement} --no-deps")


def main() -> None:
    print("=" * 60)
    print(f"{PREFIX} Safe dependency install")
    print("=" * 60)

    if not _has_module("torch"):
        print(f"{PREFIX} ERROR: torch is missing. Install or repair ComfyUI first.")
        return

    if not _has_module("torchaudio"):
        print(f"{PREFIX} ERROR: torchaudio is missing. Install or repair ComfyUI first.")
        return

    print(f"{PREFIX} torch and torchaudio are managed by ComfyUI and will not be modified.")

    _ensure("huggingface_hub", "huggingface_hub")
    _ensure("numpy", "numpy")
    _ensure("safetensors", "safetensors")
    _ensure("tokenizers", "tokenizers")
    _ensure("transformers>=4.52.1", "transformers", minimum="4.52.1")
    _ensure("torchtune==0.4.0", "torchtune")
    _ensure("moshi==0.2.2", "moshi")

    print(f"{PREFIX} Install check complete.")


if __name__ == "__main__":
    main()
