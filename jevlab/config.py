"""Paths, environment, and model names. Everything is overridable by env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

SITE_BASE = os.environ.get("JEV_SITE", "https://i-wanna-date-jev.begin-363.workers.dev")
EDITIONS = ("jev", "kev", "laya", "clef")
# Trick Kev, Trick Laya, and Trick Clef are mirrors of the game, each with its own question revisions, boards,
# and model. Each edition keeps its own database, vault, and snapshots. A mirror reads Jev's (never writes them)
# to borrow estimates.
EDITION = os.environ.get("JEV_EDITION", "jev").strip().casefold() or "jev"
if EDITION not in EDITIONS:
    raise SystemExit(f"JEV_EDITION must be one of {', '.join(EDITIONS)}, not {EDITION!r}")
MIRROR = EDITION != "jev"

JEV_DATA = Path(os.environ.get("JEVLAB_DATA", ROOT / "data"))
JEV_DB_PATH = JEV_DATA / "jev.db"
JEV_VAULT = JEV_DATA / "vault"
DATA = JEV_DATA / EDITION if MIRROR else JEV_DATA
DB_PATH = DATA / "jev.db"
SNAPSHOTS = DATA / "snapshots"
VAULT = DATA / "vault"

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
TYPESAFE_KEY = (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_KEY") or "").strip()
TYPESAFE_BASE = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai/v1")
# `typesafe/jev-latest` is rejected; the bare id is what the site sends too, and Typesafe's own API takes it.
# Each id is the `model` in that edition's site `jevRequest`. Kev is an OpenRouter /systemone model.
# Laya and Clef are scored in-process. Laya's id is its Hugging Face repo. The site asks for `clef` (27B);
# local estimates use `clef-flash`, the smallest published checkpoint.
EDITION_MODELS = {
    "jev": "jev-latest",
    "kev": "jaredpalmer/kev-4b",
    "laya": "convaiinnovations/laya",
    "clef": "clef-flash",
}
JEV_MODEL = os.environ.get("JEV_MODEL", EDITION_MODELS[EDITION])
# Hugging Face repo for `jevlab install-laya`. Independent of the current edition's oracle model.
LAYA_REPO = os.environ.get("JEV_LAYA_REPO", EDITION_MODELS["laya"])
# The model Jev's own samples were requested with, so a mirror edition can look them up in Jev's database.
JEV_SOURCE_MODEL = os.environ.get("JEV_SOURCE_MODEL", "jev-latest")
# Weights for the local Laya oracle. This stays outside the repo; `jevlab install-laya` downloads them here.
LAYA_CACHE = Path(os.environ.get("JEV_LAYA_CACHE", Path.home() / ".cache" / "jevlab" / "huggingface")).expanduser()
LAYA_DEVICE = os.environ.get("JEV_LAYA_DEVICE", "").strip()
# States per forward pass. A search round is ~90 phrases; one pass of that size exhausted RAM and swap.
LAYA_BATCH = int(os.environ.get("JEV_LAYA_BATCH", "8"))
# Hugging Face repo for `jevlab install-clef`. Clef-flash is the 9B checkpoint. `Cloudflare/clef` is the 27B
# model the site scores with.
CLEF_REPO = os.environ.get("JEV_CLEF_REPO", "Cloudflare/clef-flash")
# Weights for the local Clef oracle. Same cache root as Laya, still outside the repo.
CLEF_CACHE = Path(os.environ.get("JEV_CLEF_CACHE", Path.home() / ".cache" / "jevlab" / "huggingface")).expanduser()
CLEF_DEVICE = os.environ.get("JEV_CLEF_DEVICE", "").strip()
# auto: bf16 on a ~28GB GPU, 4-bit on a GPU with about 8GB, otherwise the checkpoint split across GPU, RAM, and disk.
CLEF_QUANT = os.environ.get("JEV_CLEF_QUANT", "auto").strip().lower() or "auto"
# Where the oracle sends /systemone. Jev pools Typesafe and OpenRouter. Kev is OpenRouter only.
# Laya and Clef load a Hugging Face checkpoint locally (`huggingface`), one forward at a time.
_LOCAL = EDITION in ("laya", "clef")
_DEFAULT_BACKENDS = "huggingface" if _LOCAL else ("openrouter" if MIRROR else "typesafe,openrouter")
ORACLE_BACKENDS = [b.strip() for b in os.environ.get("JEV_ORACLE_BACKENDS", _DEFAULT_BACKENDS).split(",") if b.strip()]
ORACLE_CONCURRENCY = int(os.environ.get("JEV_ORACLE_CONCURRENCY", "1" if _LOCAL else ("4" if MIRROR else "24")))
ORACLE_MAX_CONCURRENCY = int(os.environ.get("JEV_ORACLE_MAX_CONCURRENCY", "1" if _LOCAL else ("8" if MIRROR else "48")))
# Mirror searches first score this many lines Jev already rates well (search/triage.py) before generating any.
TRIAGE_K = int(os.environ.get("JEV_TRIAGE_K", "40" if MIRROR else "0"))
# Under load Jev's p99 is ~7s and p99.9 ~10s (2026-09-28), with a tail past 20s; slower calls give up and retry.
# Kev's host is slower. A local forward has no HTTP timeout; the value only applies if a remote backend is added.
ORACLE_TIMEOUT = float(os.environ.get("JEV_ORACLE_TIMEOUT", "60" if MIRROR else "20"))

@dataclass(frozen=True)
class GenModel:
    id: str
    tier: int  # 0 core (always on), 1 mid, 2 premium; see TIER_LEVEL
    reasoning: str  # OpenRouter reasoning effort: "none" turns thinking off, "" leaves the provider default
    max_tokens: int = 0  # raise the reply cap for models whose thinking cannot be switched off


# Generator models ranked by price ($/M tokens in/out on OpenRouter, 2026-09-26). Each tier is one bandit arm
# (`llm:core`, `llm:mid`, `llm:premium`) that takes its models round robin, so no single family has to prove
# itself first. A tier joins the search once the escalation ladder reaches TIER_LEVEL[tier].
# Thinking adds 5-50s per batch without better phrases, so it is off wherever the endpoint allows:
# GLM refuses "none", and Cohere command-a-plus fails with it (422) but works on its default.
# Everyday tier from scripts/gen_bakeoff.py (4 questions, identical prompts, average rank of top-5 oracle scores):
# gemini-3.8-flash 1.25, glm-5.3-flash 3.25, deepseek-v4-flash 3.75, gemini-2.5-flash-lite 4.0. Gemini costs ~30x
# DeepSeek per batch but a question uses only 5-20 batches (~1-2 cents). qwen3.5-flash, mistral-nemo and
# command-r7b ranked 6.5-6.75 and moved to mid; command-r7b stays for variety (it hit 0.97 where most missed).
GEN_MODEL_TABLE = [
    GenModel("google/gemini-3.8-flash", 0, "minimal"),               # 0.75/3.75
    GenModel("z-ai/glm-5.3-flash", 0, "minimal"),                    # 0.04/0.50
    GenModel("deepseek/deepseek-v4-flash", 0, "none"),               # 0.05/0.09
    GenModel("google/gemini-2.5-flash-lite", 0, "none"),             # 0.10/0.40
    GenModel("deepseek/deepseek-v4.1-flash", 1, "none"),             # 0.30/1.20
    GenModel("cohere/command-r7b-12-2024", 1, "none"),               # 0.04/0.15
    GenModel("qwen/qwen3.5-flash-02-23", 1, "none"),                 # 0.07/0.26
    GenModel("mistralai/mistral-nemo", 1, "none"),                   # 0.02/0.03
    GenModel("qwen/qwen3.6-flash", 1, "none"),                       # 0.19/1.13
    GenModel("minimax/minimax-m3", 1, "none"),                       # 0.30/1.20
    GenModel("google/gemini-3.1-flash-lite", 1, "minimal"),          # 0.25/1.50
    GenModel("openai/gpt-5.6-luna", 1, "minimal"),                   # 0.20/1.20
    GenModel("cohere/command-r-08-2024", 1, "none"),                 # 0.15/0.60, ~35s per batch
    GenModel("mistralai/mistral-small-3.2-24b-instruct", 1, "none"), # 0.09/0.25
    GenModel("qwen/qwen3.5-plus-20260420", 2, "none"),               # 0.30/1.80
    GenModel("moonshotai/kimi-k2.6", 2, "none"),                     # 0.95/4.00
    GenModel("x-ai/grok-4.3", 2, "none"),                            # 1.25/2.50
    GenModel("cohere/command-a-plus", 2, "low", 14000),              # 0.30/1.50, thinks 3-5k tokens first
    GenModel("cohere/command-a", 2, "none"),                         # 2.50/10.00
]
TIER_NAMES = {0: "core", 1: "mid", 2: "premium"}
# Ladder level at which each tier starts: L0 normal, L2 sweep, L4 reframe.
TIER_LEVEL = {0: 0, 1: 2, 2: 4}


def _gen_table() -> list[GenModel]:
    """JEV_GEN_MODELS="id:tier:reasoning,..." replaces the table (tier defaults to 0, reasoning to "none")."""
    raw = os.environ.get("JEV_GEN_MODELS", "").strip()
    if not raw:
        return GEN_MODEL_TABLE
    out = []
    for item in raw.split(","):
        parts = [p.strip() for p in item.split(":") if p.strip()]
        if parts:
            out.append(GenModel(parts[0], int(parts[1]) if len(parts) > 1 else 0,
                                parts[2] if len(parts) > 2 else "none"))
    return out


GEN_TABLE = _gen_table()
GEN_MODELS = [m.id for m in GEN_TABLE]
GEN_TIERS = {tier: [m.id for m in GEN_TABLE if m.tier == tier] for tier in sorted({m.tier for m in GEN_TABLE})}
# Single-model jobs: rewrites and synonyms use the strongest everyday model; beam next-words calls often and
# only needs single words, so it uses a fast cheap one.
GEN_MODEL = os.environ.get("JEV_GEN_MODEL", GEN_MODELS[0])
GEN_MODEL_ALT = os.environ.get("JEV_GEN_MODEL_ALT", "deepseek/deepseek-v4-flash")
PLAN_MODEL = os.environ.get("JEV_PLAN_MODEL", "google/gemini-3.1-pro-preview")
# Reasoning for models outside the table (the planner keeps its provider default).
GEN_REASONING = os.environ.get("JEV_GEN_REASONING", "none")
# Provider routing within a model: "latency" (price sort picked 20s+ hosts for mistral-nemo and Cohere),
# "price", "throughput", or "" for OpenRouter's default balancing.
PROVIDER_SORT = os.environ.get("JEV_PROVIDER_SORT", "latency")


def gen_model(model_id: str) -> GenModel | None:
    return next((m for m in GEN_TABLE if m.id == model_id), None)


# Strict Highest yes/no boards search long chains first, then compact them (search/chains.py).
# 300-word chain regime for Strict Highest. Off by default: on the boards tried it plateaued below the leader.
LONG_TARGET = int(os.environ.get("JEV_LONG_TARGET", "0"))
# Plateau mode: a Strict Highest yes/no board we already searched, whose leader holds the ceiling with a line at
# least this long. The run then races edits on dithered means and grows toward the leader's length.
PLATEAU_LEADER_WORDS = int(os.environ.get("JEV_PLATEAU_LEADER_WORDS", "40"))
# Models kept out of the long-chain round robin; each question may call them JEV_GEMINI_CALLS times, on a stall.
EXPENSIVE_MODELS = tuple(m.strip() for m in os.environ.get("JEV_EXPENSIVE_MODELS", "google/gemini-3.8-flash").split(",")
                         if m.strip())
GEMINI_CALLS = int(os.environ.get("JEV_GEMINI_CALLS", "2"))
LONG_PLAN_CALLS = int(os.environ.get("JEV_LONG_PLAN_CALLS", "2"))
LLM_TIMEOUT = float(os.environ.get("JEV_LLM_TIMEOUT", "60"))
# Stand-in model that ranks candidates by yes/no logprobs until the per-question predictor is trusted
# (`jevlab bench-proxy`: rho 0.67 vs the oracle). "" turns ProxyScreen off.
PROXY_MODEL = os.environ.get("JEV_PROXY_MODEL", "qwen/qwen3-235b-a22b-2507")
# Predictor embeddings: "local" (fastembed bge-small) or "openrouter:<model id>". Pick with `jevlab bench-embed`:
# bge-m3 ranked phrases at rho 0.815 vs 0.77 for local, and its vectors are cached in SQLite.
EMBED_BACKEND = os.environ.get("JEV_EMBED_MODEL", "openrouter:baai/bge-m3")

DATA.mkdir(parents=True, exist_ok=True)
