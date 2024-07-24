# Objective

We investigate three portfolio construction methods and improve on classical portfolio models (e.g., Markowitz).

## 1. Network Approach

## 2. Negative Skewness

## 3. Machine Learning Methods

Models seek to maximize or improve upon different measures in portfolio management: 
- Risk Adjusted Return (Sharpe Ratio, Sortino Ratio)
- Diversification
- Dynamic Asset Selection/Allocation

## Investment Universe

Daily close price of 11 US sector ETFs from 1992 to 2024: 
- `S5CONS`
- `S5ENRS`
- `S5FINL`
- `S5HLTH`
- `S5INDU`
- `S5INFT`
- `S5MATR`
- `S5TELS`
- `S5UTIL`
- `SPXT`

## 1. Network Based Approach

### Key Components of Network Based Approach

#### Correlation
- **Pearson**
- **Kendall**

#### Graph
- **G(V, E)**

#### Pruning Method
- **Complete Graph**: Every vertex is connected with all other vertices.
- **MST (Minimum Spanning Tree)**: Keeps all vertices connected, avoids cycles, and the sum of the weights is minimized.
- **PMFG (Planar Maximally Filtered Graph)**: Adds the maximum number of non-crossing edges to the MST while maintaining planarity.

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

