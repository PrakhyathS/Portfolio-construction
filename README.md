# Negative Quadratic Skewness

Negative quadratic skewness is a risk measure proposed by [Cajas (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540021) that approximates the negative component of a portfolio's skewness using a quadratic form. Minimizing the negative component of skewness allows for an increase in the portfolio's skewness, which is desirable for asset managers. The optimization of negative quadratic skewness can be obtained by solving the following problem:

$$
\begin{aligned}
\min_{x, \nu} \; & \nu \\
\text{s.t.} &  \| V^{1/2}x \| \leq \nu \\
&\sum_{i=1}^n x_i = 1 \\
& x \geq 0 \\
\end{aligned}
$$

