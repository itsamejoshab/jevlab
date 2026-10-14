# The bandit

Budget allocation is a non-stationary multi-armed bandit in `jevlab/search/scheduler.py`. Each search job is an arm. Generator models share one arm per price tier. This page is the math. The plain account is in [How a search runs](../how-it-works.md#who-gets-the-next-turn).

## Reward

After a round that spent $\mathrm{calls}$ oracle scores, improved the best fitness by $\Delta f$ (zero when the best did not move), filled or improved $\Delta e$ archive cells, and found $\mathrm{novel}$ strong lines unlike the current best:

$$
r = \frac{10 \cdot \max(\Delta f,\, 0) + 0.2 \cdot \Delta e + 0.5 \cdot \min(\mathrm{novel},\, 10)}{\max(\mathrm{calls}/100,\; 0.25)}.
$$

The denominator is the cost in units of 100 scores, floored so a tiny round is not treated as free. A novel line is one within $0.02$ of the best rounded score whose content words overlap the current top five by less than 34% Jaccard. The arm keeps the last eight rewards. Its exploit term is the mean of that window.

## Choice

Unpulled arms go first. An unpulled arm whose ladder boost is at least $4$ goes before the other unpulled arms. After that, with $T$ the number of pulls so far and $n_a$ this arm’s pulls:

$$
\mathrm{value}(a) = \Big(\bar r_a + 0.3 \sqrt{\tfrac{2 \ln (T+1)}{n_a}}\Big) \cdot \pi_a \cdot \beta^{\mathrm{plan}}_a \cdot \beta^{\mathrm{level}}_a \cdot (1 + 0.05\, U),
$$

where $U$ is uniform on $[0, 1]$.

- $\pi_a$ is a fixed prior. On Shortest yes, compress is $1.6$, single-word is $2.5$ when the leader has at most two words ($1.2$ otherwise), and boosters drop to $0.5$.
- $\beta^{\mathrm{plan}}$ is the planner’s mode. Explore favors writers and the surrogate search. Exploit favors edits, the genetic search, and GCG. Compress favors compression.
- $\beta^{\mathrm{level}}$ is the ladder. The rung’s lead job gets $4\times$. Entering the reframe also multiplies writers by $1.6$ and the surrogate search by $1.2$.

Arm state is saved per question and board, so a later run starts with the earlier runs’ evidence.
