# Word surrogate and HotFlip

Jev exposes no gradient. `WordSurrogate` in `jevlab/search/surrogate.py` is a stand-in that does. The plain account is in [How a search runs](../how-it-works.md#gcg).

## The model

Each word is embedded with the same embedder as the [phrase surrogate](surrogate.md). A phrase of $L$ words, with word vectors $e_1, \ldots, e_L$, is pooled three ways and concatenated:

$$
\psi(x) = \Big[\, \frac{1}{L}\sum_i e_i \;\Big|\; \sum_i w_i e_i \;\Big|\; L/60 \,\Big],
\qquad
w_i = \frac{i}{\sum_{j=1}^{L} j}.
$$

Later words weigh more in the second pool. A one-hidden-layer network (64 ReLU units) maps $\psi(x)$ to $\mathrm{logit}(p)$. It is trusted once its 80/20 holdout Spearman $\rho$ reaches $0.3$, which is stricter than the phrase surrogate’s $0.2$.

## The swap estimate

Because the pooling is linear in each $e_i$, the network has a gradient $g_i = \nabla_{e_i} \hat f(x)$ at every position. Replacing the word at $i$ with a vocabulary word $w$ changes the input by $e_w - e_{x_i}$. The first-order guess at the change in prediction is

$$
\Delta \hat f(i \to w) \approx (e_w - e_{x_i})^{\top} g_i.
$$

That is the HotFlip estimate. GCG keeps the top 16 vocabulary words at each position, builds one- and two-swap candidates from those lists, re-ranks the candidates with the network’s full prediction (the linear guess is only the shortlist), and sends the top of that ranking to the oracle.

The same per-position ranking orders substitution candidates in the L2 sweep when the word surrogate is trusted.
