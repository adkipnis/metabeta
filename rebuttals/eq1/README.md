# Rebuttal Equation 1 - NLL decomposition

![Equation 1](eq1.png)

The expected negative log-likelihood of the amortized posterior $p_\Pi$ decomposes into two terms: the irreducible entropy of the true posterior $p^*$ (a lower bound independent of the posterior network $\Pi$), and the expected KL divergence between $p^*$ and $p_\Pi$. Minimising the NLL over the weights of network $\Pi$ is therefore equivalent to minimising the expected KL.
