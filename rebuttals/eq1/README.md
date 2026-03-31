# Rebuttal Equation 1 - NLL decomposition

$$
\begin{aligned}
\mathbb{E}_{\boldsymbol{\vartheta}, \mathbf{D}}\!\left[ -\log p_\Pi(\boldsymbol{\vartheta}\mid \mathbf{D}) \right]
&= \mathbb{E}_{\mathbf{D}}\,\mathbb{E}_{\boldsymbol{\vartheta}|\mathbf{D}}
\!\left[ - \log p^*(\boldsymbol{\vartheta}\mid \mathbf{D})
+ \log\frac{p^*(\boldsymbol{\vartheta}\mid \mathbf{D})}{p_\Pi(\boldsymbol{\vartheta}\mid \mathbf{D})} \right] \\[6pt]
&= \mathbb{E}_{\mathbf{D}}\!\left[ H\!\left(p^*(\boldsymbol{\vartheta}\mid \mathbf{D})\right) \right]
+ \mathbb{E}_{\mathbf{D}}\!\left[ \mathrm{KL}\!\left[p^*(\boldsymbol{\vartheta}\mid \mathbf{D})\,\|\,p_\Pi(\boldsymbol{\vartheta}\mid \mathbf{D})\right] \right]
\end{aligned}
$$

The expected negative log-likelihood of the amortized posterior $p_\Pi$ decomposes into two terms: the irreducible entropy of the true posterior $p^*$ (a lower bound independent of $\Pi$), and the expected KL divergence from $p^*$ to $p_\Pi$. Minimising the NLL over $\Pi$ is therefore equivalent to minimising the expected KL — the entropy term is a constant that does not affect the optimum.
