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
| Oracle | `JEV_MODEL` | `jev-latest` (Kev: `jaredpalmer/kev-4b`) | The classifier the game scores with. Every candidate phrase is scored here. |
| Generators | `JEV_GEN_MODELS` | tiered table in `config.py` | Write candidate phrases. Each tier is one bandit arm. |
| Rewrite model | `JEV_GEN_MODEL` | first table entry | Single-model jobs: rewrites and synonyms. |
| Beam model | `JEV_GEN_MODEL_ALT` | `deepseek/deepseek-v4-flash` | High-volume next-word proposals. Should be fast and cheap. |
| Planner | `JEV_PLAN_MODEL` | `google/gemini-3.1-pro-preview` | Reads the search state on a stall and proposes directives, bans, and seeds. |
| Expensive models | `JEV_EXPENSIVE_MODELS` | `google/gemini-3.8-flash` | Kept out of the long-chain round robin; called at most `JEV_GEMINI_CALLS` times per question. |
| Proxy screen | `JEV_PROXY_MODEL` | `qwen/qwen3-235b-a22b-2507` | Ranks candidates by yes/no logprobs at cold start, before the per-question predictor is trusted. |
| Embeddings | `JEV_EMBED_MODEL` | `openrouter:baai/bge-m3` | Phrase features for the surrogate predictors. |

### Oracle backends

The oracle sends the site's own request body (each question's `jevRequest`, with the phrase as `state`) to Jev's
`/systemone` endpoint. It can pool two backends:

- `openrouter` uses `OPENROUTER_API_KEY`.
- `typesafe` uses `TYPESAFE_API_KEY` (Typesafe serves the same Jev model).

`JEV_ORACLE_BACKENDS` chooses which to use (default `typesafe,openrouter` on Jev, `openrouter` on Kev). Backends
without a key are skipped, so OpenRouter alone is enough. Calls go to the healthy backend with the most free slots,
and a backend that fails three times in a row rests for a while.

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

## Editions: Jev and Kev

`JEV_EDITION=kev` (or `--edition kev`) switches to Trick Kev, the mirror game. It changes these defaults:

| Setting | Jev | Kev |
| --- | --- | --- |
| Data directory | `data/` | `data/kev/` |
| `JEV_MODEL` | `jev-latest` | `jaredpalmer/kev-4b` |
| `JEV_ORACLE_BACKENDS` | `typesafe,openrouter` | `openrouter` |
| `JEV_ORACLE_CONCURRENCY` / max | 24 / 48 | 4 / 8 |
| `JEV_ORACLE_TIMEOUT` | 20 s | 60 s |
| `JEV_TRIAGE_K` | 0 | 40 |

Kev reads Jev's database and vault read-only (`JEV_SOURCE_MODEL` names the model Jev's samples were recorded under),
so it can start from lines that already work on Jev.

## Other settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEV_SITE` | the live site | Base URL for snapshot and publish |
| `JEV_COOKIE` | empty | Session cookie header for signed-in calls (or use `session.json`) |
| `JEVLAB_DATA` | `./data` | Database, vault, snapshots, models |
| `JEV_LLM_TIMEOUT` | 60 | Seconds per generator or planner call |
| `JEV_TRIAGE_K` | 0 (Kev 40) | Existing lines to score before generating |
| `JEV_LONG_TARGET` | 0 | Word target for the experimental long-chain regime (0 is off) |
| `JEV_PLATEAU_LEADER_WORDS` | 40 | Leader length that triggers plateau mode |
| `JEV_LONG_PLAN_CALLS` | 2 | Planner calls per question in the long-chain regime |
| `JEV_THREADS` | min(8, cores) | CPU threads for numpy, scikit-learn, and ONNX |
