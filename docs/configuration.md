# Configuration

All settings are environment variables, read once when `jevlab.config` is imported. jevlab loads `.env` from the
repo root on startup, and real environment variables take precedence. [`.env.example`](../.env.example) lists every
variable with its default; this page explains what the model roles are and when to change them.

The source of truth is [`jevlab/config.py`](../jevlab/config.py).

## The model roles

jevlab calls several different models, each for a different job. Only the first one is the "real" game model; the
rest are tools the search uses to find phrases that score well on it.

| Role | Variable | Default | What it does |
| --- | --- | --- | --- |
| Oracle | `JEV_MODEL` | `jev-latest` (Kev: `jaredpalmer/kev-4b`, Luna: `openai/gpt-6-luna-decisions`, Decider: `perplexity/pplx-decider-v1-27b`, Laya: `convaiinnovations/laya`, Clef: `clef-flash`) | The classifier the game scores with. Every candidate phrase is scored here. |
| Generators | `JEV_GEN_MODELS` | tiered table in `config.py` | Write candidate phrases. Each tier is one bandit arm. |
| Rewrite model | `JEV_GEN_MODEL` | first table entry | Single-model jobs: rewrites and synonyms. |
| Beam model | `JEV_GEN_MODEL_ALT` | `deepseek/deepseek-v4-flash` | High-volume next-word proposals. Should be fast and cheap. |
| Planner | `JEV_PLAN_MODEL` | `google/gemini-3.1-pro-preview` | Reads the search state on a stall and proposes directives, bans, and seeds. |
| Expensive models | `JEV_EXPENSIVE_MODELS` | `google/gemini-3.8-flash` | Kept out of the long-chain round robin; called at most `JEV_GEMINI_CALLS` times per question. |
| Proxy screen | `JEV_PROXY_MODEL` | `qwen/qwen3-235b-a22b-2507` | Ranks candidates by yes/no logprobs at cold start, before the per-question predictor is trusted. |
| Embeddings | `JEV_EMBED_MODEL` | `openrouter:baai/bge-m3` | Phrase features for the surrogate predictors. |

### Oracle backends

The oracle sends the site's own request body (each question's `jevRequest`, with the phrase as `state`) to Jev's
`/systemone` endpoint. It can pool two remote backends:

- `openrouter` uses `OPENROUTER_API_KEY`.
- `typesafe` uses `TYPESAFE_API_KEY` (Typesafe serves the same Jev model).

`huggingface` is a local checkpoint. It needs no API key. Laya installs with `uv sync --extra laya` and
`jevlab install-laya`. Clef installs with `uv sync --extra clef` and `jevlab install-clef` (see
[QUICKSTART.md](../QUICKSTART.md)). Weights go to `JEV_LAYA_CACHE` or `JEV_CLEF_CACHE` (both default to
`~/.cache/jevlab/huggingface`), outside the repo. `JEV_LAYA_DEVICE` and `JEV_CLEF_DEVICE` are `cpu`, `cuda`, or
`mps`. The site scores Clef with the 27B model. Local estimates use Clef-flash (`Cloudflare/clef-flash`, 9B),
the smallest published checkpoint. `JEV_CLEF_QUANT=auto` keeps it in bf16 on a GPU with about 28GB, uses 4-bit
on a GPU with about 8GB, and otherwise splits the weights across GPU, RAM, and disk. `none` forces the whole
checkpoint onto one device. Clef does not call the Cloudflare Workers AI API.

`JEV_ORACLE_BACKENDS` chooses which to use (default `typesafe,openrouter` on Jev, `openrouter` on Kev, Luna,
and Decider, `huggingface` on Laya and Clef). Remote backends without a key are skipped, so OpenRouter alone is enough for Jev
and Kev. Calls go to the healthy backend with the most free slots, and a backend that fails three times in a row
rests for a while. Laya and Clef score one phrase at a time in-process; a missing checkpoint fails with the
install command rather than downloading during a search.

`JEV_ORACLE_CONCURRENCY` is the starting number of parallel calls, and AIMD adjusts it up to
`JEV_ORACLE_MAX_CONCURRENCY`. Lower both if you hit rate limits.

### Generator tiers

The generator table lives in `GEN_MODEL_TABLE` in `config.py`. Every entry has a model id, a tier, and an OpenRouter
reasoning effort:

- **Tier 0, core**: always on. Cheap models that write good phrases.
- **Tier 1, mid**: joins at escalation level 2 (sweep), adding variety.
- **Tier 2, premium**: joins at escalation level 4 (reframe), for the hardest boards.

Within a tier, models take turns round robin, so no single model family has to prove itself first. Thinking is
turned off wherever the endpoint allows it: it adds 5-50 s per batch without writing better phrases.

To replace the table without editing code, set `JEV_GEN_MODELS` to a comma-separated list of `id:tier:reasoning`:

```bash
JEV_GEN_MODELS=google/gemini-3.8-flash:0:minimal,deepseek/deepseek-v4-flash:0:none,x-ai/grok-4.3:2:none
```

The tier defaults to 0 and reasoning to `none`. Reasoning can be `none`, `minimal`, `low`, `medium`, `high`, or empty
for the provider default. Some models refuse `none` (GLM, Cohere command-a-plus); give those `minimal` or `low`.

The default ranking came from [`scripts/gen_bakeoff.py`](../scripts/gen_bakeoff.py): the same questions and prompts
for every model, ranked by the oracle scores of each model's top five phrases. Rerun it to rank new models.

### Proxy and embeddings

Both have benchmark commands that measure agreement with the oracle on data you already have:

```bash
uv run jevlab bench-proxy                                          # default shortlist (see jevlab/bench.py)
uv run jevlab bench-proxy --models qwen/qwen3-235b-a22b-2507 deepseek/deepseek-v4-flash
uv run jevlab bench-embed --models openrouter:baai/bge-m3 local
```

`local` embeddings run fastembed's bge-small on the CPU and need the extra: `uv sync --extra embed`. OpenRouter
embeddings are cached in SQLite, so every phrase is embedded once.

Set `JEV_PROXY_MODEL=` (empty) to turn the proxy screen off.

### Provider routing

`JEV_PROVIDER_SORT` controls how OpenRouter picks a host within one model: `latency` (default), `price`,
`throughput`, or empty for OpenRouter's own balancing. Sorting by price picked slow (20 s+) hosts for some models,
which is why the default is latency.

## Cost

Rough numbers from the author's runs, at OpenRouter prices on 2026-09-26:

- A typical question uses 5-20 generator batches, about 1-2 cents with the core tier.
- Oracle calls are the bulk of the spend. `--budget` caps oracle calls per session (default 20000), and every
  sample is cached forever, so reruns on the same question get cheaper.
- The planner and premium tiers only run on stalls and escalation.

For a cheap first run, use `--budget 2000`, or `--no-llm` for a purely mechanical search with no generator calls.

## Editions: Jev, Kev, Luna, Decider, Laya, and Clef

`JEV_EDITION=kev`, `luna`, `decider`, `laya`, or `clef` (or `--edition`) switches to a mirror game. Kev, Luna, and
Decider keep the remote oracle on OpenRouter and lower concurrency. Laya and Clef score with a local Hugging Face model.

| Setting | Jev | Kev | Luna | Decider | Laya | Clef |
| --- | --- | --- | --- | --- | --- | --- |
| Data directory | `data/` | `data/kev/` | `data/luna/` | `data/decider/` | `data/laya/` | `data/clef/` |
| `JEV_MODEL` | `jev-latest` | `jaredpalmer/kev-4b` | `openai/gpt-6-luna-decisions` | `perplexity/pplx-decider-v1-27b` | `convaiinnovations/laya` | `clef-flash` |
| `JEV_ORACLE_BACKENDS` | `typesafe,openrouter` | `openrouter` | `openrouter` | `openrouter` | `huggingface` | `huggingface` |
| `JEV_ORACLE_CONCURRENCY` / max | 24 / 48 | 4 / 8 | 4 / 8 | 4 / 8 | 1 / 1 | 1 / 1 |
| `JEV_ORACLE_TIMEOUT` | 20 s | 60 s | 60 s | 60 s | unused (local) | unused (local) |
| `JEV_TRIAGE_K` | 0 | 40 | 40 | 40 | 40 | 40 |
| Weights | — | — | — | — | `~/.cache/jevlab/huggingface` | `~/.cache/jevlab/huggingface` |

A mirror reads Jev's database and vault read-only (`JEV_SOURCE_MODEL` names the model Jev's samples were recorded
under), so it can start from lines that already work on Jev. `jevlab --edition clef vault import-jev` copies the
best measured Strict line per board in as an estimate. `vault import-from --from-edition kev` does the same from
any other edition.

## Other settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEV_SITE` | the live site | Base URL for snapshot and publish |
| `JEV_COOKIE` | empty | Session cookie header for signed-in calls (or use `session.json`) |
| `JEVLAB_DATA` | `./data` | Database, vault, snapshots, models |
| `JEV_LLM_TIMEOUT` | 60 | Seconds per generator or planner call |
| `JEV_TRIAGE_K` | 0 (Kev, Laya, and Clef 40) | Existing lines to score before generating |
| `JEV_LONG_TARGET` | 0 | Word target for the experimental long-chain regime (0 is off) |
| `JEV_PLATEAU_LEADER_WORDS` | 40 | Leader length that triggers plateau mode |
| `JEV_LONG_PLAN_CALLS` | 2 | Planner calls per question in the long-chain regime |
| `JEV_THREADS` | min(8, cores) | CPU threads for numpy, scikit-learn, and ONNX |
