# Fitness and board order

The search climbs a smooth score. The board publishes a coarser one. Both live in `jevlab/objective.py`. This page is the math. The plain account is in [How a search runs](../how-it-works.md#what-the-search-is-climbing).

A phrase $x$ has $u(x)$ words (characters on Golf) and a mean goal probability $p$. When the goal answer is "no", every sample is flipped ($p \mapsto 1-p$) before either formula runs, so the lab is always maximizing. $\mathrm{round}(p)$ is the nearest hundredth, which is what the site shows.

The logit keeps a slope near the ceiling, where raw probability flattens:

$$
\mathrm{logit}(p) = \log \frac{p}{1-p}
$$

with $p$ clipped into $[0.001,\, 0.999]$.

## Strict Highest

The board orders lines by

$$
\mathrm{key}_{H}(x) = \big(\mathrm{round}(p),\; -u(x)\big).
$$

A higher rounded score always ranks first. Length breaks ties.

The search climbs

$$
f_{H}(x) = \mathrm{logit}(p) - 0.004 \cdot \max\big(0,\; u(x) - u_{\mathrm{lead}} + 1\big).
$$

$u_{\mathrm{lead}}$ is the current leader’s length. Length is free until the line is one word under that length. Past it, each extra word costs $0.004$ in logit space, which is smaller than the logit gap of a real one-point score gain near the top. With no leader yet, the penalty applies to every word.

Between $0.97$ and $0.99$ the probability moves by $0.02$, while

$$
\mathrm{logit}(0.99) - \mathrm{logit}(0.97) \approx 1.12.
$$

## Strict Shortest yes

A line qualifies when $p \ge t$. On the site the yes line is usually $0.5$. Inside the lab $t$ is that question’s published threshold, and $0.51$ when the question does not name one, so noise has a margin under the published line.

The published order is

$$
\mathrm{key}_{S}(p, u) =
\begin{cases}
(1,\; -u,\; \mathrm{round}(p)) & p \ge t \\
(0,\; 0,\; \mathrm{round}(p)) & \text{otherwise.}
\end{cases}
$$

The lab’s own ordering of scored lines is one step finer. A line whose mean is still under $t$, but which cleared $t$ on at least one sample, ranks with the qualifiers by length, and behind a same-length line whose mean does qualify:

$$
\mathrm{score}_{S} =
\begin{cases}
(1,\; -u,\; 1,\; \mathrm{round}(p)) & \text{mean qualifies} \\
(1,\; -u,\; 0,\; \mathrm{round}(p)) & \text{some sample qualifies} \\
(0,\; 0,\; 0,\; \mathrm{round}(p)) & \text{never qualified.}
\end{cases}
$$

The shaped fitness matches that ladder. With $\mathrm{gap} = \mathrm{logit}(p) - \mathrm{logit}(t)$ and $B(u) = 2 + 0.2 \cdot \max(60 - u,\, 0)$:

$$
f_{S}(x) =
\begin{cases}
B(u) + 0.1 + 0.09 \cdot \mathrm{clip}(\mathrm{gap}/4,\, 0,\, 1) & \text{mean qualifies} \\
B(u) + 0.09 \cdot \mathrm{hits}/n & \text{some samples qualify} \\
\min(\mathrm{gap},\, 0) - 0.01\, u & \text{never qualified.}
\end{cases}
$$

Each word saved is worth $0.2$, and every tie-break term stays under $0.2$. A shorter qualifying line therefore outranks a longer one. Lines that have never crossed $t$ still get a slope toward it.

## Long-chain regime

Off unless `JEV_LONG_TARGET` is set. Below the observed ceiling the length penalty shrinks to $0.0002$ per word, so lines can grow. At the ceiling only length remains:

$$
f_{\mathrm{long}}(x) =
\begin{cases}
\mathrm{logit}(c) + 1 + 0.01 \cdot \max(U - u,\, 0) & p \text{ is at the ceiling } c \\
\mathrm{logit}(p) - 0.0002 \cdot u & \text{otherwise,}
\end{cases}
$$

where $U$ is the word target.
