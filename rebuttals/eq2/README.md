# Rebuttal Equation 2 - Hierarchical Summarization with Pooling over observations

## Hierarchical Summary Network
For $d$ fixed effects, $q$ random effects and groups $i = 1, \ldots, m$ with $n_i$ observations, the local summaries are defined as

$$\mathbf{D}_i = \left[\mathbf{y}_i, \mathbf{X}_i, \mathbf{Z}_i\right] \in \mathbb R^{n_i \times (d + q)}$$
$$\mathbf{s}_i &= f_{\Sigma_l}(\mathbf{D}_i) \in \mathbb R^{1 \times h_l}$$

where $f_{\Sigma_l}$ is the local summary network with output dim $h_l$. Local summaries are stacked and fed into the global summary network:
\begin{align*}
\tilde{\mathbf{s}} &= [\mathbf{s}_i]_{i=1, \ldots, m} \in R^{m \times h_l}\\
\mathbf{s} &= f_{\Sigma_g}(\tilde{\mathbf{s}}) \in R^{1 \times h_g}
\end{align*}
where $f_{\Sigma_g}$ is the global summary network with output dim $h_g$. Before entering as context to the Posterior Network, local and global summaries are appended with local and global metadata respectively (e.g. $n_i$ resp. $m$).
