# Phrase surrogate

`PhraseSurrogate` in `jevlab/search/surrogate.py` ranks phrases the oracle has not scored yet. The plain account of the embeddings and the training loop is in [How a search runs](../how-it-works.md#embeddings).

## Features and target

A phrase becomes a vector

$$
\phi(x) = \big[\, e(x) \;\big|\; u(x) / 60 \,\big],
$$

where $e(x)$ is a sentence embedding and $u(x)$ is the word count. The length coordinate is clipped at $1$ for the cross-question model. The regression target is $\mathrm{logit}(p)$ with the probability clipped into $[0.005,\, 0.995]$, so a phrase at $0.99$ and a phrase at $0.97$ stay far apart. See [Fitness](fitness.md) for why the logit is the scale.

The fit is Bayesian ridge regression. Five-fold cross-validation holds each phrase out once, and Spearman $\rho$ is one correlation on those out-of-fold predictions together. Each fold still trains on the other 80%. The model used for ranking is then refit on every row. Bayesian ridge also returns a predictive standard deviation $\sigma(x)$.

The surrogate is trusted once $\rho \ge 0.2$. Below that, screening falls back to the logprob proxy or to a uniform draw.

## Upper confidence bound

Surrogate-guided search ranks a pool by

$$
\mathrm{UCB}(x) = \mu(x) + \kappa\, \sigma(x), \qquad \kappa = 1.5.
$$

Ordinary screening (the genetic search, local edits, and the rest) uses the same shape with $\kappa = 1$, then keeps the top 80% of its budget and draws the remaining 20% from the phrases the ranker left behind.

## Cross-question prior

`jevlab/search/global_model.py` fits one ridge regression on text framed as

```text
Q: <title> goal <goal> || <phrase>
```

The targets are offsets from each question’s mean logit, so the model learns rank inside a question. It is checked with grouped 5-fold cross-validation, holding out a whole question at a time, and it retrains after 500 newly scored phrases anywhere.

The local model and the prior are blended by how much the local holdout is trusted:

$$
\mu(x) = w \big(\mu_{\mathrm{prior}}(x) + \bar y_q\big) + (1-w)\, \mu_{\mathrm{local}}(x),
\qquad
w = \mathrm{clip}\big(1 - \rho_{\mathrm{local}} / 0.6,\; 0,\; 1\big).
$$

$\bar y_q$ is this question’s mean training logit, because the prior predicts an offset. A question with no local model is ranked entirely by the prior, provided the prior’s own $\rho$ is at least $0.2$. As the local $\rho$ climbs toward $0.6$, $w$ falls to zero.
