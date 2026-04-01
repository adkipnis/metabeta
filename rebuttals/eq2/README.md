# Rebuttal Equation 2 - Pooling

## Hierarchical Summary Network
For $d$ fixed effects, $q$ random effects and groups $i = 1, \ldots, m$ with $n_i$ observations, the local data and their summaries are defined as

$$\mathbf{D}_i = \left[\mathbf{y}_i, \mathbf{X}_i, \mathbf{Z}_i\right] \in \mathbb R^{n_i \times (d + q)}$$
$$\mathbf{s}_i = f_{\Sigma_l}(\mathbf{D}_i) \in \mathbb R^{1 \times h_l}$$

where $f_{\Sigma_l}$ is the local summary network with output dim $h_l$. Local summaries are stacked and fed into the global summary network:

$$\tilde{\mathbf{s}} = [\mathbf{s}_i]_{i=1, \ldots, m} \in R^{m \times h_l}$$
$$\mathbf{s} = f_{\Sigma_g}(\tilde{\mathbf{s}}) \in R^{1 \times h_g}$$

where $f_{\Sigma_g}$ is the global summary network with output dim $h_g$. Before entering as context to the Posterior Network, local and global summaries are appended with local and global metadata respectively (e.g. $n_i$ resp. $m$).


## Pooling at the end of the Set Transformer
Each summary network thus pools over a different level of observations: The local summarizer pools over observations per group, the global summarizer pools over groups. The permutation-invariant pooling operation is implemented with a learned CLS token.

Let $\mathbf H^{(0)} \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ be the input matrix to the set transformer (after linear embedding). The CLS token $\mathbf{c}\in \mathbf{R}^{d_{\mathrm{model}}}$ is simply concatenated as the first element in the sequence:
$$\in \mathbb{R}^{(n+1) \times d_{\mathrm{model}}} $$

After passing through $\ell$ transformer encoder blocks we get $\tilde{\mathbf H}^{(\ell)} \in \mathbb{R}^{(n+1) \times d_{\mathrm{model}}}$. The first row is extracted and passed through a linear projection. Given fixed network weights, this output is invariant to permutations of $\mathbf H^{(0)}$ along the attended axis. Intuitively $\mathbf{c}$ thus aggregates information via the multihead attention over several blocks.

$$\tilde{\mathbf H}^{(0)} = \begin{pmatrix} \mathbf c \\\ \mathbf H^{(0)}\end{pmatrix}$$
