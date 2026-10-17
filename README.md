# jevlab

Jevlab is a closed-loop, offline black-box optimizer that climbs a remote language model’s answer probability without touching the public board. It replays the site’s own scoring request against a pooled local oracle, caches every sample, and vaults a phrase only once repeated draws put its noise-aware lower bound above the live leader. A non-stationary upper-confidence bandit, with an eight-round memory and a stall-triggered escalation ladder, spends the query budget across language-model proposal, one-edit neighborhoods, clause-boundary genetic splicing, compression beams, and coordinate-ascent sweeps of a twenty-thousand-word vocabulary. Because the scorer exposes no gradient, two stand-ins are fit on our own logits: a Bayesian ridge on sentence embeddings that ranks untested phrases by upper confidence bound, and a position-pooled network whose embedding-space gradient supplies HotFlip swap estimates for a greedy-coordinate walk. A quality-diversity archive, fifteen length buckets by rhetorical frame, keeps the search off a single paraphrase basin, and a cross-question prior ranks a cold start until the local model is trusted. The same Strict line is then cross-posted onto Golf by character count and onto Casual boards that answer would take, while defense rewrites the public word list so the phrase that took the board is no longer the obvious one to mine.

**An offline search lab for [Trick Jev](https://trickjev.com).**

Search runs on this machine against a local copy of the model. Nothing hits the public board until a line is published. We search **Strict**, mainly **Strict Highest**, and also **Strict Shortest yes**. Casual and Golf are not searched. A Strict line that would take one of those boards is cross-posted, which is where the cheap wins come from.

[![jevlab home screen](docs/screenshots/lab.png)](docs/screenshots/lab.png)
[![jevlab search](docs/screenshots/search.png)](docs/screenshots/search.png)

## The search

A run scores phrases locally and remembers every score. A [scheduler](docs/how-it-works.md#who-gets-the-next-turn) spends the scoring budget on whichever job has been moving the best line. The jobs are the rows in the bottom right of the search screen.

**[Generators](docs/how-it-works.md#generators)** (`llm:core`, and later `llm:mid`, `llm:premium`). Language models write whole phrases. Each model is asked for the same few ways of speaking to Jev, and the one whose phrases score highest keeps writing until the best line stops improving. Pricier models join only after the search has already stalled.

**[Local edit](docs/how-it-works.md#local-edit).** One-word changes around a line that already scores well: swap a word, drop a word, insert a word, or move a clause. Cheap, and useful once a good line exists.

**[Compress](docs/how-it-works.md#compress).** Shortens a strong line while the score holds. It tries dropping one word, two neighboring words, or a whole clause, and keeps a cut only after the remaining phrase is scored. The word colors on the search screen are a side measurement, not the order it cuts in.

**[Genetic](docs/how-it-works.md#genetic-search).** Builds children from lines that already score well. Two parents are spliced at a clause break, or one parent is edited a few times. A small learned model throws out the weak children before they spend a real score.

**[Surrogate BO](docs/how-it-works.md#surrogate-guided-search).** Makes a large pile of local edits, asks that same learned model which ones look promising, and scores only the top of the pile. The model is trained on our own scores for this question, from [sentence embeddings](docs/how-it-works.md#embeddings).

**[GCG](docs/how-it-works.md#gcg).** Uses a second learned model, one that can say which single-word swap should help, then checks the best guesses with a real score.

When the best line goes quiet, the search steps up a [ladder](docs/how-it-works.md#the-ladder). Each rung is a heavier job, and any new best drops back to ordinary search.

**[Precision](docs/how-it-works.md#precision)** (L1). Scores the same line several times. The model’s answers wobble, and the average of those wobbles separates lines that round to the same percent.

**[Sweep](docs/how-it-works.md#sweep)** (L2). Tries a large vocabulary in every word slot and every gap, then pairs the best single edits.

**[Boosters](docs/how-it-works.md#boosters)** (L3). Short word sequences that have raised the score on other questions, stuck on as a prefix, a suffix, or a clause.

**[Beam](docs/how-it-works.md#beam)** (L4). Builds a line from nothing, one word at a time, keeping the best few at each length. Mid-tier models join at the sweep, and the most expensive models join here.

**[Single word](docs/how-it-works.md#single-word).** Strict Shortest yes only. Most of those boards are led by one word, so this scores single words and pairs directly: model ideas, other players’ words, then slices of a large vocabulary.

Some Highest boards are led by a very long line that already sits at the top score the model will report. Ordinary edits cannot see a difference there. These jobs run only in [that case](docs/how-it-works.md#plateau-jobs):

**[Extend](docs/how-it-works.md#extend)** and **[grow](docs/how-it-works.md#grow)** lengthen a line, a clause at a time or one appended word at a time, up toward the leader’s length.

**[Probe climb](docs/how-it-works.md#probe-climb)** appends a short claim for the other answer, which pulls tied lines down into a range where they separate, then climbs on that hidden ranking.

**[Scenario join](docs/how-it-works.md#scenario-join)** has a model write high-stakes scenes and joins each scene to our best tied lines.

**[Drift](docs/how-it-works.md#drift)** walks toward the neighbor whose own edits most often keep the top score. Those lines have been the ones with an edit that scores one point higher.

**[Long chains](docs/how-it-works.md#long-chains)** (`chain`, `chain_ablate`, `chain_compact`) assemble very long lines out of fragments. They are off unless `JEV_LONG_TARGET` is set.

The technical account of each job, including how parents are chosen and how children are thrown out, is in [docs/how-it-works.md](docs/how-it-works.md).

## Around the boards

**[Publish](docs/how-it-works.md#the-vault)** plays a vault line on the live site. A line is vaulted only after several local scores agree that its lower bound would beat the current leader.

**[Cross-post](docs/how-it-works.md#cross-post).** Golf is filled from the Strict vault on every publish: the same text, ranked by characters. Casual is optional on the publish screen (`Include casual proxy`). It takes the Strict Highest answer and offers it on a Casual board that answer would take, including an empty board. A Casual leader of one or two words usually keeps a tie.

**[Import](docs/how-it-works.md#import).** The publish screen can copy another edition’s best measured Strict lines in as estimates (Jev, Kev, Luna, Decider, Laya, Clef). Borrowed lines are posted without a fresh local check. Lines this edition scored itself are re-checked before they go out.

**[Defend](docs/how-it-works.md#defend).** On boards we already lead, the public word list is a copy of our winning phrase, and other players mine it. Defense posts the question’s own words, a few synonyms, then strong words from other boards, so that list stops pointing at the phrase. It skips a board whose top list is already crowded.

**[Target a player](docs/how-it-works.md#target-a-player).** On the search setup screen, picking a name selects every board that player currently leads, so a run can chase one rival.

**[Target an answer](docs/how-it-works.md#target-an-answer).** On a choice question, a run can aim at one option. The score is then that option’s probability, and the vault keeps those lines separate.

**[Live mode](docs/how-it-works.md#live-mode).** Follows the site’s live round, searches it, and can post as it goes. Fast posts the first word immediately and then every gain. Slow posts only while we are not first. The chain can be Casual (the phrase as one turn) or Strict (one word at a time).

**[Win mode](docs/how-it-works.md#win-mode).** After a line takes the lead, the run stops once it has spent a set number of further scores, instead of searching the rest of the budget.

The same lab runs the mirror games. Add `--edition kev`, `luna`, `decider`, `laya`, or `clef`. Install and the first search are in [QUICKSTART.md](QUICKSTART.md).

## Get started

```bash
uv sync
cp .env.example .env        # add your OPENROUTER_API_KEY
uv run jevlab snapshot
uv run jevlab lab --q is-cereal-a-soup
```

## More

| Doc | What it is |
| --- | --- |
| [docs/how-it-works.md](docs/how-it-works.md) | Technical account of each search job and the board tools |
| [docs/techniques/](docs/how-it-works.md#the-math) | The formulas, one page per technique |
| [QUICKSTART.md](QUICKSTART.md) | Install, the vault, publishing, and the other editions |
| [docs/commands.md](docs/commands.md) | Every command and flag |
| [docs/configuration.md](docs/configuration.md) | Model roles, costs, and environment variables |
| [scripts/README.md](scripts/README.md) | Research scripts |

## Be a good guest

Trick Jev is someone’s hobby project. Searching stays offline so the site is left alone. Snapshot occasionally, keep the default publish delays, and remember that `jevlab publish` plays real turns on a public leaderboard.

## License

[MIT](LICENSE)
