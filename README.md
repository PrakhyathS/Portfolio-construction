# Constructing and Managing Portfolios in Quantitative Wealth and Investment Management (QWIM)

## Objective

We investigate and enhance three portfolio construction methods to improve upon classical portfolio models, such as those developed by Markowitz. The aim is to refine and optimize portfolio management strategies in quantitative wealth and investment management.


## 1. Network Approach

## 2. Negative Skewness

## 3. Machine Learning Methods

Models seek to maximize or improve upon different measures in portfolio management: 
- Risk Adjusted Return (Sharpe Ratio, Sortino Ratio)
- Diversification
- Dynamic Asset Selection/Allocation

## Investment Universe

Daily close price of 10 US sector ETFs from 1992 to 2024:

| `S5CONS` | `S5ENRS` | `S5FINL` | `S5HLTH` | `S5INDU` | `S5INFT` | `S5MATR` | `S5TELS` | `S5UTIL` | `SPXT` |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|--------|


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

## Results:
- ### **Chances to Beat Benchmark**
![My Image](https://github.com/PrakhyathS/Portfolio-construction/blob/main/NetworkBasedResults/NetworkResult2.png)

- ### **1992-2024 - Diversification and Allocation on Date 2008/11/03**
<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/PrakhyathS/Portfolio-construction/blob/main/NetworkBasedResults/NetworkResult3.png" alt="Image 2" style="width: 45%; height: 250px;">
    <img src="https://github.com/PrakhyathS/Portfolio-construction/blob/main/NetworkBasedResults/NetwrokResult4.png" alt="Image 3" style="width: 45%; height: 250px;">
</div>

## Observations

- Network-based approaches generally construct more diversified portfolios compared to the GMV (Global Minimum Variance) approach.
- The MST (Minimum Spanning Tree) method consistently maintains the best diversification.

## 2. Negative Quadratic Skewness

Negative quadratic skewness is a risk measure proposed by [Cajas (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540021) that approximates the negative component of a portfolio's skewness using a quadratic form. Minimizing the negative component of skewness allows for an increase in the portfolio's skewness, which is desirable for asset managers. The optimization of negative quadratic skewness can be obtained by solving the following problem:

$$
\begin{aligned}
\min_{x, \nu} \; & \nu \\
\text{s.t.} &  \| V^{1/2}x \| \leq \nu \\
&\sum_{i=1}^n x_i = 1 \\
& x \geq 0 \\
\end{aligned}
$$

## Results:
<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/PrakhyathS/Portfolio-construction/blob/main/SkewnessResults/SkewResult1.png" alt="Image 1" style="width: 45%; height: 250px; margin: 0 10px;">
    <img src="https://github.com/PrakhyathS/Portfolio-construction/blob/main/SkewnessResults/SkewResult2.png" alt="Image 2" style="width: 45%; height: 250px; margin: 0 10px;">
</div>

- The negative quadratic skewness is a new risk measure that allows us to approximate the negative component of portfolio skewness and provides an alternative way to incorporate positive skewness into our portfolios.

- This formulation is very flexible because it can be combined with other risk measures or used to address other portfolio problems, such as risk constraints, maximization of the return-risk ratio, or achieving risk parity.


  
<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/PrakhyathS/Portfolio-construction/blob/main/SkewnessResults/SkewResult3.png" alt="Image 1" style="width: 90%; height: 300px;">
</div>

## 3. Machine Learning Method

**Primary Objective:** Dynamic asset selection and allocation with monthly rebalancing using machine learning models.

This approach heavily utilizes methods from Carta et al. ‘Ensembling and Dynamic Asset Selection for Risk-Controlled Statistical Arbitrage’ (2021). It involves the following steps and components:

### Model Algorithms
- **Random Forest**
- **Support Vector Regressors**
- **Random Decision Trees (XGBoost)**
- **ARIMA**

The fitted model predictions are averaged to produce an ensemble prediction. This ensemble is then used to calculate mean-directional accuracy, which determines trading decisions for each period.

### Features for Regression Models
- **Daily Lagged Returns**
- **Exponential Moving Average (10-day window)**
- **Stochastic Oscillator %K**
- **William’s %R**
- **Price Rate-of-Change**
- **Relative Strength Indicators**
- **Accumulation Distribution Oscillator**
- **Moving Average Convergence-Divergence**
- **Disparities (5-day and 10-day windows)**

## Results:
- ### **Monthly Returns**
![My Image](https://github.com/PrakhyathS/Portfolio-construction/blob/main/MLResults/MLResult1.png)
- The Carta portfolio, which trades `k` pairs of long-short positions, generally underperforms compared to long-only portfolios but offers lower volatility. Despite this, the long-only portfolio consistently performs better, demonstrating the efficacy of the strategy in asset selection.

- ### **Stress-Test-Returns**
![My Image](https://github.com/PrakhyathS/Portfolio-construction/blob/main/MLResults/MLResult2.png)

### Stress-Test Periods
We consider two main stress-test periods:
- **Global Financial Crisis**
- **Onset of COVID-19**

**Findings:**
- The long-only portfolio performs best during both stress-test periods.
- The long-short portfolio provides some risk control during the COVID-19 pandemic but struggles significantly during the height of the global financial crisis.

**Potential Causes and Solutions:**
- A possible cause for the long-short portfolio's failure is the investment universe.
- A potential solution could involve changing the trading rules based on specific criteria for return, which could be a topic for future research.

## Comparing Methodologies:
![My Image](https://github.com/PrakhyathS/Portfolio-construction/blob/main/MLResults/ComparisionTable.png)
### Conclusion

Each method excels in different areas:

- **Machine Learning Ensemble:** Performed best on average in terms of mean annualized return, Sharpe ratio, and Sortino ratio, delivering the best risk-adjusted return.
- **Network Analyses and Negative Quadratic Skewness:** Reduced portfolio volatility.
- **Overall Performance:** All portfolio methods outperformed both the equal-weighted and the standard Global Minimum Variance portfolios in terms of risk-adjusted returns.

**Contact**: For inquiries or further information, contact [prakhyathshivappa@gmail.com](mailto:prakhyathshivappa@gmail.com)
