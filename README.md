# jevlab

**An offline search lab for playing [Trick Jev](https://trickjev.com).**

Trick Jev poses a question. A player answers with a phrase, and Jev, a language model, reads that phrase and reports how sure it is of an answer. jevlab searches for the phrases that rank highest, on your own machine, and holds the winners until you choose to publish them on the public leaderboard.

[![jevlab home screen](docs/screenshots/lab.png)](docs/screenshots/lab.png)
[![jevlab search](docs/screenshots/search.png)](docs/screenshots/search.png)

## Strict Highest

**Strict Highest** is the main board. Two rules decide the order, and they are applied in that order:

1. **Higher confidence wins.** Jev's confidence in the target answer is rounded to the nearest percentage point. A phrase at 99% always ranks above a phrase at 98%.
2. **Fewer words breaks a tie.** When two phrases round to the same confidence, the shorter one ranks higher.

A shorter phrase never overtakes a higher score. Length matters only after the rounded confidence is equal.

```mermaid
flowchart TB
    Q["The question"] --> J["Jev reads the phrase<br/>and answers"]
    P["The phrase<br/>a chain of allowed words"] --> J
    J --> C["Confidence in the target answer<br/>rounded to the nearest 1%"]
    C --> RANK["Board order"]
    W["Word count"] --> RANK
    RANK --> R1["1. Higher confidence ranks first"]
    RANK --> R2["2. Fewer words ranks first<br/>only when confidence is tied"]
```

The same two facts decide whether a new phrase is worth keeping. The numbers below are an illustration, not a live board:

| Phrase | Confidence | Words | Where it ranks |
| --- | --- | --- | --- |
| a long argument that cereal counts as soup | 99% | 40 | Second. Same confidence as the line below, but longer. |
| soup | 99% | 1 | First. Same confidence, fewer words. |
| breakfast | 98% | 1 | Third. One point lower, so the short length does not help. |

### What a search does

A search rehearses that ranking in private. It scores phrases against a local copy of Jev and does not touch the public board until you publish.

It remembers every phrase it scores. New phrases come from two kinds of work, and a scheduler spends the scoring budget on whichever kind is actually moving the best line.

**Writing.** Language models write whole phrases, each one in a chosen way of speaking to Jev: a scene, a deduction, a direct claim, a worked example, and others. At the start of a question every model in the pool writes the same few of those ways. The model whose phrases score highest keeps writing, in every way, until the best line stops improving. The pool is asked again then. On a question that has already been searched hard, it is asked again sooner, because one model stops covering every way of phrasing.

**Editing.** Other jobs start from a line that already scores well and change it: drop a word, swap a word, cross two lines, or grow a long one. A learned stand-in ranks the large piles of edits so only the promising ones spend a real score.

A phrase is held for publishing only after several scores agree that it would outrank the current public leader. One lucky score is not enough. Putting that phrase on the live board is a separate step.

```mermaid
flowchart TB
    ASK["Ask every model for the same few ways of phrasing"] --> PICK["Keep the model whose phrases score highest"]
    PICK --> WRITE["That model keeps writing,\nswitching among ways of phrasing"]
    KNOWN["Lines that already score well"] --> EDIT["Change them: drop a word,<br/>swap a word, or join two lines"]
    WRITE --> SCORE["Score the phrase privately"]
    EDIT --> SCORE
    SCORE --> REMEMBER["Remember the score"]
    REMEMBER --> HOLD{"Repeated scores say it<br/>beats the public leader?"}
    HOLD -- yes --> PUB["Hold it until you publish"]
    REMEMBER --> QUIET{"Has the best line stopped moving?"}
    QUIET -- "still moving" --> WRITE
    QUIET -- "still moving" --> EDIT
    QUIET -- "gone quiet" --> ASK
```

The lab also plays **Strict Shortest yes**. There a phrase only has to convince Jev, just over 50%, and the shortest convincing phrase wins. The diagram above is Strict Highest.

The wording rules, the noise in Jev's answers, and the detail of the probe, the frames, and the scheduler are in [docs/how-it-works.md](docs/how-it-works.md).

## Get started

```bash
uv sync
cp .env.example .env        # add your OPENROUTER_API_KEY
uv run jevlab snapshot
uv run jevlab lab --q is-cereal-a-soup
```

See [QUICKSTART.md](QUICKSTART.md) for the full walkthrough: installation, the first search, the vault, publishing, and the Kev, Laya, and Clef editions.

## Documentation

| Doc | Contents |
| --- | --- |
| [QUICKSTART.md](QUICKSTART.md) | Install, configure, and run everything |
| [docs/how-it-works.md](docs/how-it-works.md) | How a search runs: the rules, the noise, the model probe, and the scheduler |
| [docs/commands.md](docs/commands.md) | Every command and flag |
| [docs/configuration.md](docs/configuration.md) | Model roles, swapping models, costs, Jev vs Kev vs Laya vs Clef |
| [docs/architecture.md](docs/architecture.md) | Full engine write-up, with the math, plus a source map |
| [scripts/README.md](scripts/README.md) | Research scripts and benchmarks |

## Be a good guest

Trick Jev is someone's hobby project. jevlab does its searching offline precisely so it doesn't hammer the site, so keep it that way: snapshot occasionally, keep the default publish delays, and remember that `jevlab publish` plays real turns on a public leaderboard.

## License

[MIT](LICENSE)
