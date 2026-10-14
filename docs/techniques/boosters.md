# Booster mining

Boosters are short word sequences that raise the goal probability on many questions. Applying them is described in [How a search runs](../how-it-works.md#boosters). This page is how a sequence gets its score, in `jevlab/search/boosters.py`.

## Mined lift

Mining spends no oracle calls. For each yes/no question with at least 60 scored phrases, every n-gram of one to three words (function-word-only grams dropped) is compared with the rest of that question, in logit space:

$$
\mathrm{lift}_q(g) = \overline{\mathrm{logit}}_{\,g \in x} - \overline{\mathrm{logit}}_{\,g \notin x}.
$$

A gram needs at least five phrases inside the question. Its stored lift is the mean of $\mathrm{lift}_q$ over the questions where it appears. It is kept when that mean is positive, it appears in at least three questions, and at least fifteen phrases contain it. The ranking used to decide which sixty to keep weights a gram that shows up widely:

$$
\mathrm{rank}(g) = \overline{\mathrm{lift}}(g) \cdot \sqrt{N},
$$

$N$ is the number of questions the gram appears in. The number written down as `lift` is the unweighted mean, not this rank.

## Optimised triggers

A separate pass spends oracle calls. Coordinate ascent edits a $k$-word trigger, tried as a prefix and as a suffix, to maximise the mean logit gain across a pool of questions. Each question contributes our best line there as the base. An optimised trigger is stored with source `optimised` and outranks a mined gram when the library is applied.

When the library is used, a trigger’s win rate is smoothed the same way as a [hold rate](resolution.md#hold-rate), $(w+1)/(t+2)$, and the library sorts by positive lift, then optimised-over-mined, then that rate, then raw lift.
