# Scripts

Research scripts that sit on top of the `jevlab` package. They are not part of the CLI, and most of them were written
to answer one question about one board. Each file's docstring explains what it tests. Run them from the repo root:

```bash
uv run python scripts/<name>.py [args]
```

## `gen_bakeoff.py`

A head-to-head of generator models: the same questions and the same prompt for every model, with every phrase scored
by the oracle. The default generator tiers in `jevlab/config.py` come from this script.

```bash
uv run python scripts/gen_bakeoff.py [slug ...]
```

## `experiments/`

Arguments are positional. `SLUG` is a question slug and `LEAD` is the board leader's word count.

| Script | Usage | Question it answers |
| --- | --- | --- |
| `append_walk.py` | `[BOARDS] [BUDGET] [...]` | Does a deep walk of end-appends from a strong line reach the next level? |
| `bag_climb.py` | `SLUG LEAD [BUDGET] [...]` | Generate long lines until one reads the next level, then prune below the leader |
| `bag_walk.py` | `SLUG LEAD [N_RANDOM] [N_LLM_CALLS] [...]` | Build long lines from the leader's per-word tallies, then prune |
| `join_rate.py` | `[PAIRS]` | How often joining two dissimilar top lines reaches 0.99 |
| `llm_walk.py` | `BOARDS BUDGET [...]` | End-append walk where an LLM proposes the next words |
| `long_paragraphs.py` | `[MODELS]` | Do long fluent paragraphs from strong LLMs reach 0.99? |
| `raw_reply.py` | | Print one full `/systemone` reply and the request shape |
| `rescore_old.py` | `[CUTOFF] [PER]` | Did the oracle drift? Rescore old 0.99 lines |
| `rule_polish.py` | `[PER_LINE]` | Single-word appends and inserts on rule-stack lines |
| `rule_stacks.py` | `[BOARDS]` | Rule-stack scaffolds that redefine the question's terms |
| `scenario_shift.py` | `SLUG LEAD [ROUNDS]` | Does a different scenario lift a tuned line past 0.98? |
| `side_channel.py` | | Does a choice twin of a yes/no question give finer resolution? |
| `site_long.py` | `SLUG LEAD [REROLLS] [MIN_WORDS] [...]` | How does the live site score a long strict line? (posts to the site) |
| `site_screen.py` | `SLUG LEAD [LINES] [REROLLS] [...]` | Screen long lines on the live site (posts to the site) |
| `stack_joins.py` | | Do joins, repeats, and stacks of 0.98 lines reach 0.99? |
| `vault_hits.py` | `SLUG LEAD [MODE]` | Shrink confirmed 0.99 lines and queue the shortest in the vault |

`BOARDS` is a comma-separated list of the board names defined at the top of each script. The two `site_*` scripts
play turns on the live site and need a signed-in session.
