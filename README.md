### Project Description
This project investigates whether a dynamic portfolio based on market regimes outperforms an equal-weight ETF portfolio. We perform the analysis by computing the Sharpe ratio, Maximum Drawdown, and final cumulative return (percentage) of each portfolio to evaluate performance.

### Asset Universe
- SPY => (Equities) => SPDR S&P500 ETF Trust Equity tracking S&P500 index (US large-cap stocks) 
- GLD => (Gold) => SPDR Gold Shares Commodity ETF tracking gold prices.
- USO => (Oil) => United States Oil Fund Commodiy ETF tracking crude oil prices (WTI)
- BND => (Bonds) => Vanguard Total Bond Market ETF Bond providing broad exposure to US.

### HMM model
In order to define market regime signals we decide to adopt a data drive approach using HMM model to define hiddne state transitions among three hidden states (hidden state value has been defined using data driven approach and equal to 5 for optimal. Despite the answer, we where forced by HMM to reduce to 3 hidden states due to size of window and so our interest in keeping the window of size 252 trading days. The HMM model has been trained on momentum and volatility of returns all computed manually with scratch functions.

### Momentum_window=60
values used in industry:
- 20 days => short term market regime
- 60 days => balance between short term and smoothness (preferred choice)
- 120 days => smoother trends, less sensitivity to short term noise

### Volatility_window=20
values used in industry:
- 20 days => monthly
- 60 days => stable
volatility changes quickly, 20 is standard for regime switching models

### HMM for reproducibility
EM algorithm in HMMs starts with random initialization of:
- Initial state probabilities
- Transition probabilities
- Emission parameters

Without fixing **random_state**, each run starts from a different configuration, yielding different final parameters and log-likelihood values, by using **random_state=42** we ensure reproducibility.

### Hidden State Selection Problem
The optimal number of hidden states was initially identified as 5 based on the Bayesian Information Criterion (BIC), which balances model fit and complexity.
However, during implementation, the model produced numerical instabilities (NaN values in the transition or initial probability matrices), indicating insufficient data support for estimating all parameters in a 5-state model.

To ensure model stability and convergence, the number of hidden states was reduced to 3, which provided consistent estimation of transition and emission probabilities while preserving the main regime dynamics.

### How n_params in BIC computation has been determined:
For a Gaussian HMM with \(K\) hidden states and \(D\) observed features, the number of free parameters is computed as:

### Number of Parameters in the Model
### Number of Parameters in the Model

n_params = K * [D(D + 1) / 2] + K * D + K * (K - 1) + (K - 1)

Where:
- D = number of features  
- K = number of states  

Breakdown:
- K * [D(D + 1) / 2] → covariance parameters (full covariance matrix per state)  
- K * D → mean parameters (mean vector per state)  
- K * (K - 1) → transition probabilities (each row sums to 1, so K - 1 free parameters per row)  
- (K - 1) → initial state probabilities (sum to 1, so K - 1 free parameters)

### PCA Decomposition
We apply PCA decomposition, since variables are each other correlated and HMM works better with a small amount of variables. By performing PCA we reduce dimensionality (less variables) and we keep only components up to explain 90% of original variance within the dataframe. Since PCA produces linearly uncorrelated components, the covariance matrix of the transformed data becomes (approximately) diagonal.
Therefore, using covariance_type="diag" in the HMM is justified, as it reduces the number of parameters, eases computation, and typically leads to faster convergence.

### Logging for info showing
we use logging build in package to manage INFO showing on software ongoing and provide a more polish look to output.

### Double rolling window approach
We employ a rolling window of 252 days to generate a time series of market hidden states. To initialize the process, the first 252 days are used as the initial window, resulting in the removal of the first 252 rows from the final dataset.

For each subsequent day `t`, we use the most recent 252 days of data to compute state-specific parameters: a mean vector and a covariance matrix for each of the three hidden states. The model's predicted hidden state for day `t` then selects the corresponding mean and covariance pair. These selected parameters serve as the inputs for the optimization of portfolio methods, which produces a set of optimal portfolio weights. This procedure is repeated daily, generating a daily series of portfolio allocations.

### Dictionary Logic implemented
The dictionary structure stores the mean and covariance matrices estimated for each hidden state considering the values of the given rolling window (to simulate a real life estimation and avoid data leakage pitfal).
The key of the dictionary corresponds to the date of the last observation in the window, ensuring temporal alignment between the estimated parameters (weights) and the predicted hidden state (and so with mean and covariance matrix) for \(t+1\)

### Portfolio optimization weights
I performed the comparison between MVO, GMV and EW, we are interested in using a Equal Weight portfolio as benchmark, so a portfolio in which all assets are weighted the same, a Mean Variance Optimization method and a Global Minima Variance portfolio. The weights are computed using dedicated functions and results are saved on above dictionary and saved as .joblib object to store in memory and access it without recomputation.

### Returns computation
Due to construction of dictionary that stores hidden states and mean and covariance matrices, and original dataframe with returns it was necessary to use three keys of indeces from dates.
- t_0 => the current date => used to access return 
- t_1 => the day before => used to access current portfolio weights
- t_2 => t-2 => the day before yesterday = used to access previous portfolio weights and implement the cost turnover.

### Cost turnover
Since we are interested in simulating a real life dynamic optimizer we decide to implement a cost_factor of 0.005% per unit of turnover, to compute the turnover we use the summatory of the different absolute values of the weights absolute difference between consequent periods.

### Metrics computation
In order to compare performances of portfolios we compute:
- Sharpe Ratio
- Max DrawDown Ratio
- Cumulative return generated (%)

### Conclusion
| Portfolio | Sharpe Ratio | Max Drawdown | Cumulative Return [%] |
|-----------|-------------|--------------|---------------------|
| ret_mvo   | -0.02441    | -4.071       | 31.16               |
| ret_gmv   | 0.01078     | -4.897       | 38.16               |
| ret_ew    | -0.12461    | -4.736       | 14.32               |

As expected, the dynamic portfolios outperformed the equal-weight portfolio, achieving an average cumulative return of ~30% over the period 2015–2024, compared to 14% for the equal-weight portfolio.

Among the dynamic portfolios, GMV optimization outperformed MVO, with a +7% higher cumulative return and a positive Sharpe ratio, indicating outperformance relative to the risk-free asset on a volatility-adjusted basis.

Additionally, GMV exhibited a lower Maximum Drawdown, reflecting smaller losses before recovery from peaks.

Conclusion: Considering cumulative returns, Sharpe ratio, and drawdown, GMV is the preferred optimization method, with MVO and equal-weight portfolios as alternative strategies that generate positive returns over time.