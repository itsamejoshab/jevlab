# Command reference

Generated from `jevlab <command> --help`. Every command also accepts `--edition jev|kev|laya|clef` anywhere on the line.
Run through uv as `uv run jevlab ...`, or plain `jevlab ...` inside the activated venv.

Running `jevlab` with no command opens the TUI home screen.

## Global

```text
usage: jevlab [-h] [--edition {jev,kev,laya,clef}]
              {tui,install-laya,install-clef,snapshot,calibrate,lab,search,triage,score,vault,publish,crosspost,errors,boosters,bench-embed,bench-proxy,train-global} ...

Offline Jev search lab

positional arguments:
  {tui,install-laya,install-clef,snapshot,calibrate,lab,search,triage,score,vault,publish,crosspost,errors,boosters,bench-embed,bench-proxy,train-global}
    tui                 home screen: choose Search or Publish (default when no command is given)
    install-laya        download Laya weights from Hugging Face into ~/.cache/jevlab (outside the repo)
    install-clef        download Clef weights from Hugging Face into ~/.cache/jevlab (outside the repo)
    snapshot            pull questions, boards, word impacts, and our attempts into data/jev.db
    calibrate           replay site-scored phrases through the oracle and report parity
    lab                 interactive TUI search for one yes/no or choice question (strict highest or shortest
                        yes)
    search              headless search for one yes/no or choice question (strict highest or shortest yes)
    triage              score the most promising existing lines (Jev's vault and history, estimated vault
                        lines) with this edition's oracle, without searching
    score               score phrases with the oracle
    vault               list saved winners, queue them for publishing, or drop them; import-from copies
                        another edition's best measured Strict lines in as estimates
    publish             submit queued vault entries to the live site (separate batch loop)
    crosspost           queue Strict vault lines that beat a Golf leader into the golf vault
    errors              failed network calls by service and host (OpenRouter vs the Jev site)
    boosters            universal boosters: mine from samples, optimise across questions, list
    bench-embed         compare embedding models for the phrase predictor (holdout rho)
    bench-proxy         test logprob chat models as stand-ins for Jev (rho vs oracle)
    train-global        train the cross-question predictor on every scored phrase

options:
  -h, --help            show this help message and exit
  --edition {jev,kev,laya,clef}
                        game edition: jev (data/), kev (data/kev/), laya (data/laya/), or clef (data/clef/);
                        read before anything else loads
```

## Laya weights

### `jevlab install-laya`

Downloads `convaiinnovations/laya` into `~/.cache/jevlab/huggingface` (override with `JEV_LAYA_CACHE`). Needs
`uv sync --extra laya` first. The download stays outside the git repo. Search does not fetch weights on its own.

```text
usage: jevlab install-laya [-h]

options:
  -h, --help  show this help message and exit
```

## Clef weights

### `jevlab install-clef`

Downloads `Cloudflare/clef-flash` (9B) into `~/.cache/jevlab/huggingface` (override with `JEV_CLEF_CACHE`). Needs
`uv sync --extra clef` first. The download stays outside the git repo. Search does not fetch weights on its own,
and scoring does not call the Cloudflare Workers AI API. The site itself scores with the 27B model; local
estimates use flash. `JEV_CLEF_QUANT=auto` uses bf16 on a GPU with about 28GB, 4-bit on a GPU with about 8GB,
and otherwise splits the weights across GPU, RAM, and disk.

```text
usage: jevlab install-clef [-h]

options:
  -h, --help  show this help message and exit
```

## Data: snapshot and calibration

### `jevlab snapshot`

```text
usage: jevlab snapshot [-h] [--q [Q ...]] [--modes [MODES ...]] [--workers WORKERS]

options:
  -h, --help           show this help message and exit
  --q [Q ...]          limit to these slugs or play URLs
  --modes [MODES ...]  strict casual golf emoji (default all)
  --workers WORKERS
```

### `jevlab calibrate`

```text
usage: jevlab calibrate [-h] [--q [Q ...]] [--per-question PER_QUESTION] [--n N]

options:
  -h, --help            show this help message and exit
  --q [Q ...]           limit to these slugs
  --per-question PER_QUESTION
  --n N                 oracle samples per phrase
```

## Search

### `jevlab tui`

```text
usage: jevlab tui [-h]

options:
  -h, --help  show this help message and exit
```

### `jevlab lab`

```text
usage: jevlab lab [-h] --q Q [--mode {strict}] [--board {highest,shortest}] [--budget BUDGET] [--no-llm]
                  [--seed SEED] [--no-escalate] [--max-level MAX_LEVEL] [--target TARGET]

options:
  -h, --help            show this help message and exit
  --q Q                 question slug or play URL
  --mode {strict}       v1 supports strict only
  --board {highest,shortest}
                        game mode board to compete on
  --budget BUDGET       max oracle calls this session
  --no-llm              skip LLM generators and planner
  --seed SEED
  --no-escalate         move on at a plateau instead of escalating
  --max-level MAX_LEVEL
                        highest escalation level (0-4)
  --target TARGET       choice questions: aim at this answer (score = P(answer))
```

### `jevlab search`

```text
usage: jevlab search [-h] --q Q [--mode {strict}] [--board {highest,shortest}] [--budget BUDGET] [--no-llm]
                     [--seed SEED] [--no-escalate] [--max-level MAX_LEVEL] [--target TARGET] [--all-targets]
                     [--win-after N] [--transfer-k K] [--force FORCE] [--plateau PLATEAU]

options:
  -h, --help            show this help message and exit
  --q Q                 question slug or play URL
  --mode {strict}       v1 supports strict only
  --board {highest,shortest}
                        game mode board to compete on
  --budget BUDGET       max oracle calls this session
  --no-llm              skip LLM generators and planner
  --seed SEED
  --no-escalate         move on at a plateau instead of escalating
  --max-level MAX_LEVEL
                        highest escalation level (0-4)
  --target TARGET       choice questions: aim at this answer (score = P(answer))
  --all-targets         choice questions: run once per answer, one after another
  --win-after N         win mode: once a line beats the leader, stop after N more oracle calls
  --transfer-k K        score K existing lines (Jev's best, estimated vault lines) before generating;
                        default JEV_TRIAGE_K (40 on Kev, Laya, and Clef, 0 on Jev)
  --force FORCE         run only this strategy
  --plateau PLATEAU     flat rounds per level (0 = run to budget)
```

### `jevlab triage`

```text
usage: jevlab triage [-h] [--q [Q ...]] [--board {highest,shortest}] [--target TARGET] [--k K] [--n N]
                     [--confirm-n CONFIRM_N] [--queue]

options:
  -h, --help            show this help message and exit
  --q [Q ...]           question slugs or play URLs (default: every searchable question)
  --board {highest,shortest}
                        board to judge the lines on
  --target TARGET       choice questions: lines aimed at this answer
  --k K                 lines per question
  --n N                 oracle samples per line on the first pass
  --confirm-n CONFIRM_N
                        samples for lines whose first pass comes close
  --queue               queue the best winner per board (if none is queued)
```

### `jevlab score`

```text
usage: jevlab score [-h] --q Q [--n N] [--strict] phrases [phrases ...]

positional arguments:
  phrases

options:
  -h, --help  show this help message and exit
  --q Q
  --n N
  --strict    normalize to strict-chain words first
```

## Vault and publishing

### `jevlab vault`

```text
usage: jevlab vault [-h] [--from-edition {jev,kev,laya,clef}] [--q Q] [--phrase PHRASE] [--all]
                    [--status {candidate,queued,published,failed,rejected,dropped}] [--target TARGET]
                    [--board {highest,shortest}] [--mode {strict,golf}]
                    [{list,queue,unqueue,drop,import-jev,import-from}]

positional arguments:
  {list,queue,unqueue,drop,import-jev,import-from}

options:
  -h, --help            show this help message and exit
  --from-edition {jev,kev,laya,clef}
                        import-from: edition whose vault to read
  --q Q                 question slug or play URL
  --phrase PHRASE       exact phrase to queue/drop (default: the best candidate)
  --all                 queue every winning candidate for the question
  --status {candidate,queued,published,failed,rejected,dropped}
  --target TARGET       choice questions: only lines aimed at this answer (queue/unqueue/drop default:
                        whole-question lines)
  --board {highest,shortest}
                        game mode board (list: default every board; queue/unqueue/drop: default highest)
  --mode {strict,golf}  play mode (list: default every mode; queue/unqueue/drop: default strict)
```

### `jevlab publish`

```text
usage: jevlab publish [-h] [--q [Q ...]] [--dry-run] [--rerolls REROLLS] [--no-verify] [--delay DELAY]
                      [--board {highest,shortest}] [--mode {strict,golf}] [--no-crosspost]

options:
  -h, --help            show this help message and exit
  --q [Q ...]           only these slugs
  --dry-run             verify boards and oracle, but send no turns
  --rerolls REROLLS     re-score the final word N times even after a win; the board keeps the best
  --no-verify           skip the oracle re-check
  --delay DELAY         seconds between site turns
  --board {highest,shortest}
                        only this game mode board (default every board)
  --mode {strict,golf}  only this play mode (default every mode)
  --no-crosspost        do not follow each landed Strict line with its Golf cross-posts
```

### `jevlab crosspost`

```text
usage: jevlab crosspost [-h] [--q [Q ...]] [--board {highest,shortest}] [--dry-run]

options:
  -h, --help            show this help message and exit
  --q [Q ...]           only these slugs
  --board {highest,shortest}
                        only this Golf board (default both)
  --dry-run             list the picks without queueing them
```

## Learned models and benchmarks

### `jevlab boosters`

```text
usage: jevlab boosters [-h] [--goal [{yes,no} ...]] [--k K] [--questions QUESTIONS] [--budget BUDGET]
                       [--top TOP]
                       [{list,mine,optimise}]

positional arguments:
  {list,mine,optimise}

options:
  -h, --help            show this help message and exit
  --goal [{yes,no} ...]
  --k K                 trigger length in words (optimise)
  --questions QUESTIONS
                        questions to optimise over
  --budget BUDGET       oracle calls per goal (optimise)
  --top TOP
```

### `jevlab train-global`

```text
usage: jevlab train-global [-h] [--limit LIMIT]

options:
  -h, --help     show this help message and exit
  --limit LIMIT  cap on training rows (0 = all)
```

### `jevlab bench-embed`

```text
usage: jevlab bench-embed [-h] [--models [MODELS ...]] [--phrases PHRASES]

options:
  -h, --help            show this help message and exit
  --models [MODELS ...]
                        OpenRouter embedding model ids (default: a shortlist)
  --phrases PHRASES
```

### `jevlab bench-proxy`

```text
usage: jevlab bench-proxy [-h] [--models [MODELS ...]] [--phrases PHRASES]

options:
  -h, --help            show this help message and exit
  --models [MODELS ...]
                        OpenRouter chat model ids (default: a shortlist)
  --phrases PHRASES
```

## Diagnostics

### `jevlab errors`

```text
usage: jevlab errors [-h] [--minutes MINUTES] [--tail TAIL]

options:
  -h, --help         show this help message and exit
  --minutes MINUTES  look back this far (default 60)
  --tail TAIL        also print the last N failures
```
