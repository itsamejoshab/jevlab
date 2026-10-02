# How jevlab works

This is the technical summary that used to open the README. A plain-language account of a Strict Highest search is in the [README](../README.md). The full write-up, with the math and diagrams, is in [architecture.md](architecture.md).

## The challenges with understanding Jev

- **Black box.** Jev exposes no gradients, no logits, and no internals. We only get the final answer probability.
- **Quantized.** Every answer is rounded to 0.01. Near the top of the range, most improvements are smaller than one step.
- **Stochastic.** The same phrase returns different values on different calls (per-call σ ≈ 0.009).
- **Discrete and constrained.** The search space is sequences of Strict-legal words (Latin letters and digits, strict casing, ≤16 characters per word, ≤60 words, no fragments of choice answer names).
- **Lexicographic objective.** Boards rank by rounded score first and length second. Which is primary depends on the board.

## How the engine works

A single engine serves both boards (**Strict Highest** and **Strict Shortest yes**). A **multi-armed bandit**
allocates oracle budget across heterogeneous **proposal operators**: LLM writers, evolutionary search,
surrogate-guided Bayesian optimization, gradient-free coordinate ascent, and gradient-estimated token swaps.
Candidates are pre-screened by a **cascade of learned surrogates**. A **statistical racing layer** extracts
sub-quantum signal from the oracle's own noise. When progress stalls, an **escalation ladder** moves to progressively
more exhaustive tactics.

On **Strict Highest**, the board orders lines lexicographically by rounded probability, then fewer words:

$$
\text{key}_{\text{H}}(x) = \big(\operatorname{round}(p),\; -u(x)\big).
$$

A higher rounded score always wins. Length only breaks ties.
