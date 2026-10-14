# Reading past the rounding

The site reports probabilities rounded to the nearest hundredth. Several parts of the lab recover a finer value from repeated samples. The plain account is in [How a search runs](../how-it-works.md#precision).

## Noise

A call is treated as a noisy value that is then rounded. Calibration measured a per-call standard deviation of about $0.009$. The spread shrinks toward the ends of the range. Pooled over every line with at least five samples, by extremity $\max(p, 1-p)$:

| $\max(p, 1-p)$ | under 0.6 | under 0.7 | under 0.8 | under 0.9 | under 0.95 | 0.95 and up |
| --- | --- | --- | --- | --- | --- | --- |
| $\sigma$ per sample | 0.013 | 0.012 | 0.010 | 0.008 | 0.0055 | 0.0045 |

Once a line has eight or more samples, its own measured spread replaces the pooled value, and it is not allowed to go below that pooled value. The standard error of a mean of $n$ samples is

$$
\mathrm{SE} \approx \frac{\sigma(p)}{\sqrt{n}}.
$$

Near $0.98$, eight samples resolve the mean to a few thousandths, under one rounding step.

## Accepting a step

Precision climbing accepts a neighbor $b$ over its parent $a$ when the gain clears one combined standard error:

$$
\hat p_b - \hat p_a > \sqrt{\frac{\sigma_a^2}{n_a} + \frac{\sigma_b^2}{n_b}}.
$$

The winner is sampled again and has to keep at least half that margin. Successive halving spends the samples: everyone is topped up to 2, the worse half is dropped, the survivors go to 4, then to 8.

## The vault bound

A line is saved when its lower bound would beat the leader under the board’s own key:

$$
\mathrm{LCB}(x) = \hat p - \frac{\max(\hat\sigma,\, 0.01)}{\sqrt{n}},
\qquad
\text{win} \iff \mathrm{key}\big(\mathrm{LCB}(x),\, u(x)\big) > \mathrm{key}(\mathrm{leader}).
$$

The floor of $0.01$ on $\hat\sigma$ stops a line with identical samples from publishing on a single lucky cluster. The leader is the best other player in the last snapshot.

On Shortest yes the site keeps a player’s best roll. A line shorter than the leader that cleared the yes line on at least one of five or more samples can be vaulted on that best roll, flagged as a bet with its hit count, even when the lower bound does not qualify.

## Hold rate

At the ceiling every sample of a good line is identical, so the mean carries nothing. What still varies is the neighborhood. Drift scores a candidate by how often its one-word edits keep the parent’s rounded score, with Laplace smoothing

$$
\frac{h+1}{k+2},
$$

where $h$ of $k$ probed edits held. The walk moves to the neighbor with the highest hold rate.
