"""Local Laya oracle: the Hugging Face checkpoint, scored in-process.

The weights live in the Hugging Face cache under ``JEV_LAYA_CACHE`` (default
``~/.cache/jevlab/huggingface``), which is outside the git repo. ``jevlab install-laya``
downloads them. Search loads that cache and does not fetch on a miss.
"""

from __future__ import annotations

import threading

from .config import LAYA_CACHE, LAYA_DEVICE, LAYA_REPO

REPO = LAYA_REPO
# The checkpoint repo also ships sibling models. These are the files one English agent needs.
WEIGHT_PATTERNS = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/*",
    "encoder/*",
)

RUNTIME_MSG = (
    "Laya scores on a local Hugging Face model. Install the runtime with `uv sync --extra laya`, "
    "then download the weights with `jevlab install-laya`."
)


class LayaError(RuntimeError):
    pass


def cache_dir():
    """Directory that holds the checkpoint. Never a path inside the repo unless JEV_LAYA_CACHE says so."""
    return LAYA_CACHE


def weights_message() -> str:
    return (
        f"Laya weights are not in {cache_dir()}. Run `jevlab install-laya` "
        "(after `uv sync --extra laya`). That download stays outside the git repo."
    )


_agent = None
_lock = threading.Lock()


def _snapshot(download: bool) -> str:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError as error:
        raise LayaError(RUNTIME_MSG) from error
    if download:
        try:
            cache_dir().mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise LayaError(f"Cannot create the Laya cache at {cache_dir()}: {error}") from error
    try:
        return snapshot_download(
            REPO,
            cache_dir=str(cache_dir()),
            allow_patterns=list(WEIGHT_PATTERNS),
            local_files_only=not download,
        )
    except LocalEntryNotFoundError as error:
        raise LayaError(weights_message()) from error
    except OSError as error:
        raise LayaError(f"Cannot use the Laya cache at {cache_dir()}: {error}") from error


def install() -> str:
    """Download the checkpoint into the cache. Returns the snapshot directory."""
    return _snapshot(download=True)


def load_agent():
    """The process-wide agent. Raises LayaError when the runtime or the weights are missing."""
    global _agent
    if _agent is not None:
        return _agent
    with _lock:
        if _agent is None:
            try:
                import laya
            except ImportError as error:
                raise LayaError(RUNTIME_MSG) from error
            path = _snapshot(download=False)
            _agent = laya.load(path, device=LAYA_DEVICE or None)
        return _agent


def predict_many(states: list[str], questions: dict) -> list[dict]:
    """One system-one result per state. A single state uses system_one; several share a forward pass."""
    agent = load_agent()
    with _lock:
        if len(states) == 1:
            return [agent.system_one(states[0], questions)]
        return list(agent.predict_batch(states, questions))
