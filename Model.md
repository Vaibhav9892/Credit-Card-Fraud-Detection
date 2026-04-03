# Bayesian Fraud Model

We model the unknown fraud probability (theta) and the observed number of fraud cases (F) as follows:

$$
\begin{align*}
F &\sim \text{Binomial}(N, \theta) \\
\theta &\sim \text{Beta}(\alpha, \beta) \\
\alpha &= 1 \\
\beta &= 500
\end{align*}
$$

##### Press Ctrl+Shift+V to open the markdown preview