# Rebuttal Equation 2 - Hierarchical Summarization with Pooling over observations

Each summary network thus pools over a different level of observations: The local summarizer pools over observations per group, the global summarizer pools over groups. The permutation-invariant pooling operation is implemented with a learned CLS token.
\\ \\
Let $\mathbf H^{(0)} \in \mathbb{R}^{n \times d_{\mathrm{model}}}$ be the input matrix to the set transformer (after linear embedding). The CLS token $\mathbf{c}\in \mathbf{R}^{d_{\mathrm{model}}}$ is simply concatenated as the first element in the sequence:
$$\tilde{\mathbf H}^{(0)} =  \begin{pmatrix}
\mathbf c \\
\mathbf H^{(0)}
\end{pmatrix}\in \mathbb{R}^{(n+1) \times d_{\mathrm{model}}}$$
After passing through $\ell$ transformer encoder blocks we get $\tilde{\mathbf H}^{(\ell)} \in \mathbb{R}^{(n+1) \times d_{\mathrm{model}}}$. The first row is extracted and passed through a linear projection. Given fixed network weights, this output is invariant to permutations of $\mathbf H^{(0)}$ along the attended axis. Intuitively $\mathbf{c}$ thus aggregates information via the multihead attention over several blocks.


