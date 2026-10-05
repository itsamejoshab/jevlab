"""Local Clef oracle: the Hugging Face checkpoint, scored in-process.

The weights live in the Hugging Face cache under ``JEV_CLEF_CACHE`` (default
``~/.cache/jevlab/huggingface``), which is outside the git repo. ``jevlab install-clef``
downloads them. Search loads that cache and does not fetch on a miss.

The site scores with the 27B model. Local estimates use Clef-flash (``Cloudflare/clef-flash``, 9B),
the smallest published checkpoint, and do not call the Cloudflare Workers AI API. ``auto`` loads
that checkpoint in 4-bit on a GPU that can hold it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

from .config import CLEF_CACHE, CLEF_DEVICE, CLEF_QUANT, CLEF_REPO, JEV_MODEL

REPO = CLEF_REPO
# Clef-flash bf16 is about 19GB. A GPU at or above this holds it without quantizing.
FULL_GIB = 28
# Below this, 4-bit does not fit and the full checkpoint is split across GPU, RAM, and disk.
FOURBIT_GIB = 8

RUNTIME_MSG = (
    "Clef scores on a local Hugging Face model. Install the runtime with `uv sync --extra clef`, "
    "then download the weights with `jevlab install-clef`."
)


class ClefError(RuntimeError):
    pass


def cache_dir():
    """Directory that holds the checkpoint. Never a path inside the repo unless JEV_CLEF_CACHE says so."""
    return CLEF_CACHE


def weights_message() -> str:
    return (
        f"Clef weights are not in {cache_dir()}. Run `jevlab install-clef` "
        "(after `uv sync --extra clef`). That download stays outside the git repo."
    )


def _cuda_total_gib() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def resolve_quant(device: str) -> str:
    """``auto`` keeps bf16 on a GPU that can hold the checkpoint, 4-bit when that fits, else shards the full weights."""
    choice = CLEF_QUANT
    if choice not in {"auto", "4bit", "offload", "none"}:
        raise ClefError(f"JEV_CLEF_QUANT must be auto, 4bit, offload, or none, not {choice!r}")
    if choice != "auto":
        return choice
    if device == "cuda":
        total = _cuda_total_gib()
        if total >= FULL_GIB:
            return "none"
        if total >= FOURBIT_GIB:
            return "4bit"
    return "offload"


def prepare(device: str) -> str:
    """The quant mode to load, or an error when 4-bit was asked for off CUDA."""
    quant = resolve_quant(device)
    if quant == "4bit" and device != "cuda":
        raise ClefError(
            f"4-bit Clef runs on CUDA. This device is {device}. "
            "Set JEV_CLEF_QUANT=none to load the full checkpoint, or JEV_CLEF_DEVICE=cuda."
        )
    return quant


def resolve_device() -> str:
    if CLEF_DEVICE:
        return CLEF_DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


_agent = None
_lock = threading.Lock()


def _snapshot(download: bool) -> str:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
    except ImportError as error:
        raise ClefError(RUNTIME_MSG) from error
    if download:
        try:
            cache_dir().mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ClefError(f"Cannot create the Clef cache at {cache_dir()}: {error}") from error
    try:
        return snapshot_download(
            REPO,
            cache_dir=str(cache_dir()),
            local_files_only=not download,
        )
    except LocalEntryNotFoundError as error:
        raise ClefError(weights_message()) from error
    except OSError as error:
        raise ClefError(f"Cannot use the Clef cache at {cache_dir()}: {error}") from error


def install() -> str:
    """Download the checkpoint into the cache. Returns the snapshot directory."""
    return _snapshot(download=True)


def _joint(path: Path):
    """The release's ``joint_schema_model``, loaded from the snapshot rather than vendored."""
    name = "jevlab_clef_joint_schema_model"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    file = path / "joint_schema_model.py"
    if not file.is_file():
        raise ClefError(weights_message())
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ClefError(f"Cannot load Clef's joint schema model from {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ram_gib() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 16.0
    return pages * size / (1024 ** 3)


def _cpu_budget_bytes() -> int:
    """RAM the sharded load may occupy, leaving the rest of the machine alone."""
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        available = int(_ram_gib() * 0.5 * 1024 ** 3)
    return max(available - 6 * 1024 ** 3, 6 * 1024 ** 3)


class _TextProcessor:
    """Tokenizer stand-in. Phrase scoring is text-only, so the image and video processor stays unloaded."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _text_processor(path: Path):
    from transformers import AutoTokenizer

    return _TextProcessor(AutoTokenizer.from_pretrained(path))


def _finish(module, path: Path, backbone, dtype):
    """Attach the joint head on the embedding's device and a text tokenizer."""
    import json

    from safetensors.torch import load_file

    backbone.config.use_cache = False
    head_config = json.loads((path / "joint_head_config.json").read_text())
    head = module.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
    embed = backbone.get_output_embeddings()
    # CPU and disk offload leave the parameter on the meta device until the module runs.
    hook = getattr(embed, "_hf_hook", None)
    pre_forward = getattr(hook, "pre_forward", None)
    if pre_forward is not None:
        pre_forward(embed)
    if embed.weight.device.type == "meta":
        raise ClefError(
            "Clef's output embedding was offloaded to disk, so the joint head cannot score. "
            "Free some RAM and try again."
        )
    head = head.to(device=embed.weight.device, dtype=dtype).eval()
    return module.ClefModel(backbone, _aligned_head(head)).eval(), _text_processor(path)


def _load_full(module, path: Path, device: str):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    try:
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ClefError(f"Cannot load Clef on {device}: {error}") from error
    return _finish(module, path, backbone, torch.bfloat16)


def _offload_map(path: Path, max_memory: dict) -> dict:
    """Keep the embedding, the final norm, and the last layer on CPU so the joint head can read them.

    The rest is filled onto the GPU, then RAM, then disk. A disk-offloaded embedding is a meta tensor,
    and the head cannot score from that.
    """
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        empty = Qwen3_5ForConditionalGeneration(config)
    layer = empty.model.language_model.layers[0]
    last = f"layers.{len(empty.model.language_model.layers) - 1}"
    device_map = infer_auto_device_map(
        empty,
        max_memory=max_memory,
        dtype=torch.bfloat16,
        no_split_module_classes=[type(layer).__name__],
    )
    for name in list(device_map):
        if (
            name == "lm_head"
            or name.startswith("lm_head.")
            or name.endswith("embed_tokens")
            or name.endswith("language_model.norm")
            or name.endswith(last)
        ):
            device_map[name] = "cpu"
    return device_map


def _load_offload(module, path: Path, device: str):
    """Full bf16 checkpoint, split across the GPU, RAM, and a disk folder outside the repo."""
    import shutil

    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    offload = cache_dir() / "clef-offload"
    try:
        if offload.exists():
            shutil.rmtree(offload)
        offload.mkdir(parents=True)
    except OSError as error:
        raise ClefError(f"Cannot create the Clef offload folder at {offload}: {error}") from error
    max_memory: dict = {"cpu": _cpu_budget_bytes()}
    if device == "cuda" and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        max_memory[0] = max(free - 2 * 1024 ** 3, 1 * 1024 ** 3)
    try:
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map=_offload_map(path, max_memory),
            max_memory=max_memory,
            offload_folder=str(offload),
            low_cpu_mem_usage=True,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ClefError(f"Cannot load Clef across devices: {error}") from error
    return _finish(module, path, backbone, torch.bfloat16)


def _aligned_head(head):
    """Run the joint head on the embedding's device.

    A 4-bit load splits the backbone across GPU and CPU. The head reads hidden states from the
    last layer and token vectors from the output embedding, so those three have to meet.
    """
    import torch

    class Align(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight):
            device = output_embedding_weight.device
            return self.head(
                hidden_states.to(device),
                input_ids.to(device),
                attention_mask.to(device),
                records,
                output_embedding_weight,
            )

    return Align()


def _load_4bit(module, path: Path):
    import torch
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise ClefError(RUNTIME_MSG) from error
    compute = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    try:
        # The whole 4-bit model has to stay on the GPU. Bitsandbytes rejects a split onto CPU.
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            quantization_config=quant,
            device_map={"": 0},
            dtype=compute,
            low_cpu_mem_usage=True,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise ClefError(f"Cannot load Clef in 4-bit: {error}") from error
    return _finish(module, path, backbone, compute)


def load_agent():
    """The process-wide model, processor, and ``systemone``. Raises ClefError when setup is missing."""
    global _agent
    if _agent is not None:
        return _agent
    with _lock:
        if _agent is None:
            try:
                import torch  # noqa: F401
            except ImportError as error:
                raise ClefError(RUNTIME_MSG) from error
            path = Path(_snapshot(download=False))
            module = _joint(path)
            device = resolve_device()
            quant = prepare(device)
            if quant == "4bit":
                print(
                    "Clef is scoring locally in 4-bit. Set JEV_CLEF_QUANT=offload to use the full checkpoint.",
                    file=sys.stderr,
                )
                model, processor = _load_4bit(module, path)
            elif quant == "offload":
                print(
                    "Clef is scoring locally from the full checkpoint, split across GPU, CPU, and disk.",
                    file=sys.stderr,
                )
                model, processor = _load_offload(module, path, device)
            else:
                try:
                    model, processor = _load_full(module, path, device)
                except (ImportError, OSError, RuntimeError, ValueError) as error:
                    raise ClefError(f"Cannot load Clef on {device}: {error}") from error
            _agent = (model, processor, module.systemone)
        return _agent


def predict_many(states: list[str], questions: dict) -> list[dict]:
    """One system-one result per state. Forwards run one at a time so a 27B load stays in memory."""
    model, processor, systemone = load_agent()
    bodies = []
    with _lock:
        for state in states:
            try:
                bodies.append(systemone(model, processor, {
                    "model": JEV_MODEL,
                    "state": state,
                    "questions": questions,
                }))
            except ValueError as error:
                raise ClefError(str(error)) from error
    return bodies
