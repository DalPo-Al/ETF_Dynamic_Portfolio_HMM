# Dynamic Portfolio Optimization under Market Regimes (HMM-Based Approach)

## Project Description
This project investigates whether a dynamic portfolio based on market regimes outperforms an equal-weight ETF portfolio. The analysis evaluates performance using the following metrics:

- Sharpe Ratio  
- Maximum Drawdown  
- Final Cumulative Return (%)

These metrics are computed for each portfolio strategy to provide a comprehensive risk-adjusted performance comparison.

---

## Asset Universe
The analysis is conducted on the following ETF universe:

- **SPY** → (Equities) → SPDR S&P 500 ETF Trust  
  Tracks the S&P 500 index (US large-cap equities)

- **GLD** → (Gold) → SPDR Gold Shares  
  Commodity ETF tracking gold prices

- **USO** → (Oil) → United States Oil Fund  
  Commodity ETF tracking crude oil (WTI)

- **BND** → (Bonds) → Vanguard Total Bond Market ETF  
  Broad exposure to the US bond market

---

## Hidden Markov Model (HMM) for Market Regimes

Market regimes are identified using a data-driven Hidden Markov Model (HMM), which captures hidden state transitions across time.

- Number of hidden states: initially tested with **K = 5 (optimal via BIC)**  
- Final implementation: **K = 3 states**

Although BIC suggested 5 states as optimal, numerical instability (NaN values in transition and initial probability matrices) emerged due to insufficient data support under the rolling window constraint (252 trading days). Therefore, the model was reduced to 3 hidden states to ensure stability and convergence.

The HMM is trained on:
- Momentum (computed manually)
- Volatility of returns (computed from scratch functions)

---

## Feature Engineering

### Momentum Window = 60
Common industry benchmarks:
- 20 days → short-term regime sensitivity  
- 60 days → balanced approach (preferred)  
- 120 days → long-term smoothing, lower noise sensitivity  

### Volatility Window = 20
Common interpretations:
- 20 days → monthly volatility (standard)  
- 60 days → smoother regime estimation  

Volatility reacts quickly to market changes; thus, 20 days is typically used in regime-switching models.

---

## Reproducibility in HMM

The EM algorithm used in HMMs depends on random initialization of:
- Initial state probabilities  
- Transition matrix  
- Emission parameters  

Without fixing `random_state`, each run produces different results (parameters and log-likelihood). To ensure reproducibility, we set:

- `random_state = 42`

---

## Hidden State Selection Problem

The optimal number of hidden states was initially selected using the Bayesian Information Criterion (BIC), which balances model fit and complexity.

However:
- K = 5 led to numerical instability  
- Causes: insufficient data support within rolling estimation window  

Final decision:
- K = 3 chosen for stable estimation  
- Preserves main regime structure while ensuring convergence of transition/emission parameters  

---

## Number of Parameters in the HMM (BIC)

For a Gaussian HMM with \( K \) hidden states and \( D \) observed features, the number of parameters is:

\[
n_{\text{params}} = K \cdot \frac{D(D + 1)}{2} + K \cdot D + K \cdot (K - 1) + (K - 1)
\]

Where:
- \( D \) = number of features  
- \( K \) = number of hidden states  

### Breakdown:
- \( K \cdot \frac{D(D + 1)}{2} \) → covariance matrices (full covariance per state)  
- \( K \cdot D \) → mean vectors per state  
- \( K \cdot (K - 1) \) → transition probabilities (rows sum to 1)  
- \( (K - 1) \) → initial state probabilities  

---

## PCA Decomposition

Principal Component Analysis (PCA) is applied because:
- Input variables are highly correlated  
- HMM performs better on low-dimensional, decorrelated inputs  

We retain enough components to explain **90% of the variance**.

Since PCA produces linearly uncorrelated components:
- Covariance matrix becomes approximately diagonal  

Thus:
- `covariance_type = "diag"` is justified  
- Benefits:
  - Reduced parameter space  
  - Faster convergence  
  - Improved numerical stability  

---

## Logging System

The built-in Python `logging` module is used to:
- Monitor execution flow  
- Provide structured INFO-level outputs  
- Improve readability and debugging during runtime  

---

## Rolling Window Framework (252 Days)

A rolling window approach is used to simulate realistic, non-leaky estimation:

- Initial window: 252 trading days  
- First 252 rows are excluded from final dataset  

For each day \( t \):
- The previous 252 days are used to estimate:
  - Mean vector per state  
  - Covariance matrix per state  

The predicted hidden state determines which parameters are selected for portfolio optimization.

This process is repeated daily, generating a dynamic time series of portfolio allocations.

---

## Dictionary-Based Parameter Storage

A dictionary structure stores:
- Mean vectors per state  
- Covariance matrices per state  

Key design:
- Key = last date of rolling window  

This ensures:
- Temporal alignment between:
  - Estimated parameters  
  - Predicted hidden state  
  - Portfolio optimization inputs  

It prevents data leakage by construction.

---

## Portfolio Optimization Methods

The following strategies are compared:

- Equal Weight (EW) → benchmark  
- Mean-Variance Optimization (MVO)  
- Global Minimum Variance (GMV)  

Weights are computed using dedicated optimization functions and stored in a dictionary.

To improve efficiency:
- Results are serialized using `joblib`  
- Avoids recomputation across sessions  

---

## Serialization (Joblib)

Joblib is used for:
- Efficient serialization/deserialization of Python objects  

Definitions:
- Serialization → convert Python objects into byte streams  
- Deserialization → reconstruct objects from stored bytes  

Use case in this project:
- Store dictionary of optimized weights  
- Enable fast reload without recomputation  

---

## Returns Computation Logic

Due to the rolling structure, three temporal indices are used:

- \( t_0 \) → current date → returns  
- \( t_1 \) → previous day → current portfolio weights  
- \( t_2 \) → two days back → previous weights (for turnover calculation)  

This structure ensures consistency between:
- Portfolio weights  
- Returns  
- Transaction costs  

---

## Transaction Costs (Turnover Model)

To simulate realistic trading:

- Cost factor: **0.005% per unit of turnover**

Turnover is defined as:

\[
\text{Turnover} = \sum |w_t - w_{t-1}|
\]

This captures:
- Rebalancing intensity  
- Trading friction effects  

---

## Performance Metrics

Portfolio performance is evaluated using:

- Sharpe Ratio  
- Maximum Drawdown  
- Cumulative Return (%)  

These metrics capture:
- Risk-adjusted returns  
- Worst-case loss exposure  
- Total profitability  

---

## Results Summary

| Portfolio | Sharpe Ratio | Max Drawdown | Cumulative Return [%] |
|----------|--------------|--------------|------------------------|
| ret_mvo  | -0.02441     | -4.071       | 31.16                  |
| ret_gmv  | 0.01078      | -4.897       | 38.16                  |
| ret_ew   | -0.12461     | -4.736       | 14.32                  |

---

## Conclusion

Dynamic portfolio strategies outperform the equal-weight benchmark over the period **2015–2024**.

Key findings:
- Average cumulative return of dynamic strategies ≈ **30%**
- Equal-weight portfolio returns ≈ **14%**

Among dynamic approaches:
- **GMV outperforms MVO**
  - Higher cumulative return (+7%)
  - Positive Sharpe ratio (risk-adjusted outperformance)
  - Lower Maximum Drawdown (more stable downside behavior)

### Final Insight
Considering return, risk-adjusted performance, and drawdown behavior, **GMV is the most robust portfolio strategy in this framework**, while MVO and Equal Weight remain viable but inferior benchmarks.

---
