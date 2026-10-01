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

## 7. Trick Kev and Trick Laya

The site has two mirror games, Trick Kev and Trick Laya, each with its own model and boards. Add `--edition kev`
or `--edition laya` to any command, or set `JEV_EDITION` in `.env`. Each mirror keeps its data in `data/kev/` or
`data/laya/` and reads (never writes) Jev's data to borrow estimates.

```bash
uv run jevlab --edition laya snapshot
uv run jevlab --edition laya vault import-jev --queue   # copy Jev lines in as estimates
uv run jevlab --edition laya search --q is-cereal-a-soup
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

`data/`, `.env`, and `session.json` are gitignored. Move the data elsewhere with `JEVLAB_DATA=/path`.

## Next

- [docs/commands.md](docs/commands.md): every command and flag
- [docs/architecture.md](docs/architecture.md): how the search engine works
- [docs/configuration.md](docs/configuration.md): models, costs, and tuning
