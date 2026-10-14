# How jevlab works

This is the technical account behind the [README](../README.md). The formulas, the operator-by-operator write-up, and the source map are in [architecture.md](architecture.md).

## Why Jev is hard to read

- **Black box.** Jev exposes no gradients, no logits, and no internals. We only get the final answer probability.
- **Quantized.** Every answer is rounded to 0.01. Near the top of the range, most improvements are smaller than one step.
- **Stochastic.** The same phrase returns different values on different calls (per-call σ ≈ 0.009).
- **Discrete and constrained.** The search space is sequences of Strict-legal words.
- **Lexicographic objective.** Boards rank by rounded score first and length second. Which fact is primary depends on the board.

A phrase is a Strict chain: Latin letters and digits, no punctuation, each word all lowercase, ALL CAPS, or Capitalized, at most 16 characters per word and 60 words per phrase. Choice questions also reject a word that is a fragment of an answer name. The checker is `jevlab/rules/strict.py`, and it runs before anything is scored.

## The two boards

One engine plays both boards. `jevlab/objective.py` is the only place that defines "better."

On **Strict Highest** the board orders lines by rounded probability, then fewer words:

$$
\text{key}_{\text{H}}(x) = \big(\operatorname{round}(p),\; -u(x)\big).
$$

A higher rounded score always wins. Length only breaks ties.

On **Strict Shortest yes** a line has to qualify first: $P(\text{goal}) > 0.5$ on the site, and at least 0.51 inside the lab so noise cannot drop it under. Among qualifying lines, fewer words win, and the rounded score breaks the remaining ties. Most of those boards are led by one or two words, so the search spends its effort on the short end. The full key, and the shaped fitness the search climbs, are in [architecture §1](architecture.md#1-the-problem-formally) and [§4](architecture.md#4-fitness-shaping).

When the goal answer is "no", every probability is flipped ($p \mapsto 1-p$) so the rest of the lab is always maximizing.

## What one round does

A round is one pull of a multi-armed bandit (`jevlab/search/scheduler.py`). The bandit picks a proposal operator. The operator writes or edits phrases, a screen drops the pile down to what the budget can score, and `evaluate()` checks the Strict rules, the ban and pin lists, and the copy guard before the oracle is called. Samples are cached in SQLite, so a phrase is paid for once. The archive stores the phrase, its samples, which operator wrote it, and which frame it used. The bandit is then credited for any gain in the best line, any new region of the archive, and any strong line that does not merely paraphrase the current best.

The operators fall into two groups. Language models write whole phrases. Everything else edits a phrase the archive already trusts: one-word edits, deletion, crossover, a learned ranking of a large variant pool, and the slower sweeps the ladder turns on when the ordinary jobs go quiet. The full list is [architecture §7](architecture.md#7-proposal-operators).

## How a model gets the question

The generator pool is `JEV_GEN_MODELS`, or the default table in `jevlab/config.py`. Each entry has a tier. Tier 0 is in the pool from the start. Tier 1 joins when the ladder reaches the sweep, and tier 2 when it reaches the reframe. List the models cheapest first.

Before the first batch, and again when the current writer has gone stale, every unlocked model writes the same short batch: eight phrases spread across five frames (`rename`, `scene`, `syllogism`, `direct`, `wild`). Each model's score is the mean of its best three lines. The highest mean keeps the question. A model within 0.02 of that mean loses to whichever is listed earlier, which is why the cheap models come first. The probe's own phrases are real candidates and stay in the archive.

That model then writes every later batch, whichever frames the batch drew. It is replaced when any of these happen:

- It has written eight batches and the search has gone six rounds without a better line.
- The question is already well searched (at least half the frames have eight scored lines) and the hold is shorter: four batches and four flat rounds.
- The model returns two empty batches in a row. It sits out, and the next batch probes again.
- The ladder unlocks a higher tier, so the new models get a turn in the probe.

The decision lives in `jevlab/search/probe.py`. Rewrites and single-word synonym calls stay on `JEV_GEN_MODEL` and `JEV_GEN_MODEL_ALT`. They are not part of the probe.

## How a batch chooses its frames

The frame catalog is `ARCHETYPE_GUIDE` in `jevlab/search/prompts.py`. It is the set of rhetorical moves that change what this kind of classifier answers: a renamed subject, a scene, a speaker, a deduction, a presupposed fact, a worked example, a quoted policy, and the rest. The archive files each phrase under the frame that produced it, plus `other` when a line has no frame.

A normal batch shows the model three frames. The draw is weighted by the best score each frame has reached, so a frame that is working gets more turns. On a fresh start the weight runs the other way, toward frames that have barely been tried. Once every frame in the batch already has eight scored lines, one slot moves to a frame that does not, so a hot frame cannot starve the catalog. The first few Highest batches still reserve one slot for a deduction (`syllogism`). Shortest boards do not.

Each prompt shows the model the question, the leader's score and length (the leader's text stays hidden), the lab's own best and worst lines, and the ceiling each frame has reached. The model is told that Jev reads the phrase literally: a stated fact, a worked example, or one clear negation lands, and a double negative or an implied condition does not. The next batch is written while the oracle scores the current one.

## What is kept, and what can be published

The archive keeps every scored phrase. On top of it, a MAP-Elites grid keeps the best phrase in each cell of length-bucket × frame, so the genetic search and the variant pool draw from different shapes of line, not thirty paraphrases of one idea. Filling a new cell is paid by the bandit even before it beats the global best. See [architecture §5](architecture.md#5-the-archive-and-map-elites-diversity).

A large edit list is ranked before it is scored. Once a per-question model is trusted, it orders the pool and the lab scores the top of that ranking plus a random slice, so the model cannot hide its own blind spots. Before that model exists, a cheap stand-in reads yes/no logprobs. See [architecture §8](architecture.md#8-the-screening-cascade).

Publishing is stricter than remembering. After each round the top lines are scored again until they have at least five samples. A line is vaulted when its lower confidence bound would beat the public leader under the board's own key, so a few lucky rolls do not qualify. The leader is read from the last snapshot, with our own rows left out. `jevlab publish` is the step that plays the vaulted line on the live board. The gate is [architecture §11](architecture.md#11-promotion-verification-and-the-vault).

## When the search goes quiet

A round that does not produce a new best line counts as flat. After twelve flat rounds at the start, and six on the later rungs, the ladder steps up. Any new best drops it back to ordinary search.

| Level | What leads |
| --- | --- |
| L0 | The bandit, over writers and ordinary edits |
| L1 | A hill-climb of one-word edits, accepting a step only when the mean clears the noise |
| L2 | A coordinate sweep: try other words in each position. Mid-tier models join the next probe |
| L3 | Short triggers that have raised the goal probability across many questions, on yes/no boards |
| L4 | A beam that rebuilds the line. Premium models join the next probe |

The level reached without a win is saved, and the next run on that question resumes there. Run memory also keeps which operators have paid off, which lines have been compressed, and which frames earlier runs already pushed as far as they go. A later run treats those frames as exhausted and steers the writers toward the ones still open. See [architecture §10](architecture.md#10-the-escalation-ladder) and [§13](architecture.md#13-cross-run-memory-and-fresh-starts).
