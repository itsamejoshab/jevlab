# Quickstart

From a fresh clone to a running search in about five minutes.

## 1. Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or plain `pip`
- An [OpenRouter](https://openrouter.ai/keys) API key with a few dollars of credit

## 2. Install

```bash
git clone <this repo> jevlab
cd jevlab
uv sync                     # creates .venv and installs jevlab plus its dependencies
# optional: local CPU embeddings instead of OpenRouter ones
# uv sync --extra embed
# optional: local Laya or Clef classifier (weights are a separate download; see section 7)
# uv sync --extra laya
# uv sync --extra clef
```

With pip instead: `python -m venv .venv && . .venv/bin/activate && pip install -e .`

Every command below is shown as `uv run jevlab ...`. If you activated the venv, plain `jevlab ...` works too.

## 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY`. That is the only required value. Every model role (the Jev oracle,
the phrase generators, the planner, the proxy screen, and embeddings) has its own commented section in `.env.example`
if you want to swap models later. The details are in [docs/configuration.md](docs/configuration.md).

## 4. First run

```bash
uv run jevlab snapshot        # pull questions, leaderboards, and word stats from the live site into data/jev.db
uv run jevlab calibrate       # optional: check the local oracle agrees with site scores (needs a signed-in snapshot)
```

Pick a question slug from the snapshot output (or from a `/play?q=<slug>` URL on the site), then search:

```bash
uv run jevlab lab --q is-cereal-a-soup                        # interactive TUI, Strict Highest board
uv run jevlab search --q is-cereal-a-soup --board shortest    # headless, Strict Shortest yes board
uv run jevlab search --q is-cereal-a-soup --budget 5000       # smaller oracle budget for a cheap test run
```

Running `uv run jevlab` with no command opens the home screen, where you can pick Search or Publish.

To score your own phrases directly:

```bash
uv run jevlab score --q is-cereal-a-soup "cereal is soup" "milk makes cereal a cold soup"
```

## 5. Look at results

Lines that beat the current board leader are saved to the vault after verification.

```bash
uv run jevlab vault list                   # every saved line, grouped by question and board
uv run jevlab errors                       # recent failed network calls (OpenRouter vs the site)
```

## 6. Publish to the live site (optional)

Publishing plays your vault lines on the real leaderboard, so it needs a signed-in session:

1. Sign in on the site in your browser.
2. Copy the `__Secure-better-auth.session_token` cookie (and `__Secure-better-auth.session_data` if present).
3. Either set `JEV_COOKIE="__Secure-better-auth.session_token=...; __Secure-better-auth.session_data=..."` in
   `.env`, or create `session.json` in the repo root:

   ```json
   {"cookies": {"__Secure-better-auth.session_token": "..."}}
   ```

Then:

```bash
uv run jevlab vault queue --q is-cereal-a-soup    # queue the best line for publishing
uv run jevlab publish --dry-run                   # check boards and oracle, send nothing
uv run jevlab publish                             # send the queued lines
```

Please be a good guest. This is someone's hobby game. Keep the default delays, and don't flood the boards.

## 7. Trick Kev, Trick Laya, and Trick Clef

The site has three mirror games, Trick Kev, Trick Laya, and Trick Clef, each with its own model and boards. Add
`--edition kev`, `--edition laya`, or `--edition clef` to any command, or set `JEV_EDITION` in `.env`. Each mirror
keeps its data in `data/kev/`, `data/laya/`, or `data/clef/` and reads (never writes) Jev's data to borrow estimates.

Kev's classifier is `jaredpalmer/kev-4b` on OpenRouter, so the same API key scores it:

```bash
uv run jevlab --edition kev snapshot
uv run jevlab --edition kev vault import-jev --queue
uv run jevlab --edition kev search --q is-cereal-a-soup
```

Laya (`convaiinnovations/laya`) is not on OpenRouter. The classifier runs locally from a Hugging Face checkpoint.
The weights go to `~/.cache/jevlab/huggingface`, outside the git repo. Install the runtime, then download them once:

```bash
uv sync --extra laya
uv run jevlab install-laya
```

`JEV_LAYA_CACHE` moves that directory. `JEV_LAYA_DEVICE` is `cpu`, `cuda`, or `mps` (empty lets PyTorch pick).
Search still uses OpenRouter for the phrase generators, the planner, and embeddings, so the API key stays
required. Only the classifier is local, and it does not download on its own if the weights are missing.

```bash
uv run jevlab --edition laya snapshot
uv run jevlab --edition laya vault import-jev --queue   # copy Jev lines in as estimates
uv run jevlab --edition laya search --q is-cereal-a-soup
```

Clef on the site is the 27B model. Local estimates use Clef-flash (`Cloudflare/clef-flash`, about 19GB), the
smallest published checkpoint, so scoring stays fast. The classifier does not call the Cloudflare Workers AI
API. Weights go under `~/.cache/jevlab/huggingface` (or `JEV_CLEF_CACHE`).

```bash
uv sync --extra clef
uv run jevlab install-clef
```

`JEV_CLEF_DEVICE` is `cpu`, `cuda`, or `mps`. `JEV_CLEF_QUANT=auto` loads flash in bf16 when the GPU has about
28GB, uses 4-bit when the GPU has about 8GB (a 12GB card), and otherwise splits the weights across the GPU, RAM,
and disk. Search does not download the weights on its own. Flash estimates will not match the site's 27B scores
exactly.

```bash
uv run jevlab --edition clef snapshot
uv run jevlab --edition clef vault import-jev --queue
uv run jevlab --edition clef search --q is-cereal-a-soup
```

## 8. Tests

```bash
uv sync --group dev
uv run pytest
uv run ruff check
```

## Where things live

| Path | Contents |
| --- | --- |
| `data/jev.db` | SQLite: snapshots, boards, cached oracle samples, run memory |
| `data/vault/` | Verified winning lines per question |
| `data/snapshots/` | Raw JSON from each `jevlab snapshot` |
| `data/models/` | Trained cross-question predictor (`jevlab train-global`) |
| `data/kev/` | The same layout for the Kev edition |
| `data/laya/` | The same layout for the Laya edition |
| `data/clef/` | The same layout for the Clef edition |
| `~/.cache/jevlab/huggingface` | Laya and Clef weights (`JEV_LAYA_CACHE`, `JEV_CLEF_CACHE`) |

`data/`, `.env`, and `session.json` are gitignored. Move the data elsewhere with `JEVLAB_DATA=/path`. The Laya
and Clef weights never land in the repo; they stay in the cache above.

## Next

- [docs/commands.md](docs/commands.md): every command and flag
- [docs/how-it-works.md](docs/how-it-works.md): short technical summary of the search
- [docs/architecture.md](docs/architecture.md): full engine write-up, with the math
- [docs/configuration.md](docs/configuration.md): models, costs, and tuning
