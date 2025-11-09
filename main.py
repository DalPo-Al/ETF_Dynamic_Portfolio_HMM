from src.data_fetch import fetch_data
from src.feature_engineering import feature_engineering
from src.hmm import hmm_model
import logging
from src.mean_covariance import comp_mean_covariance, optimization, returns_computation, plotting, metrics_computation

#logging configuration
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S"
)
#variables
tickers=["SPY", "BND", 'GLD','USO']
start_date='2013-01-01'
end_date='2023-09-01'
momentum_window=60
volatility_window=20
try:
    #data featching
    logging.info('DATA_FETCHING started...')
    fetch_data(ticker=tickers, start=start_date, end=end_date)
    logging.info("DATA_FETCHING completed :)")
    #feature engineering
    logging.info('FEATURE_ENGINEERING started...')
    feature_engineering(momentum_window=momentum_window, volatility_window=volatility_window)
    logging.info("FEATURE_ENGINEERING completed :)")
    #hidden state extraction
    logging.info("HIDDEN_STATE_EXTRACTION started...")
    hmm_model()
    logging.info("HIDDEN_STATE_EXTRACTION completed :)")
    #covariance matrix and mean for hidden
    logging.info("COVARIANCE_MATRIX_AND_MEAN_EXTRACTION started...")
    comp_mean_covariance()
    logging.info("COVARIANCE_MATRIX_AND_MEAN_EXTRACTION completed :)")
    #compute portfolio weights
    logging.info("WEIGHTS_OPTIMIZATION started...")
    optimization()
    logging.info("WEIGHTS_OPTIMIZATION completed :)")
    #cumulative returns computation
    logging.info("CUMULATIVE_RETURNS_COMPUTATION started...")
    returns_computation()
    logging.info("CUMULATIVE_RETURNS_COMPUTATION completed :)")
    #plotting cumulative returns
    logging.info("PLOTTING started...")
    plotting()
    logging.info("PLOTTING completed :)")
    #metrics computation
    logging.info("METRICS_COMPUTATION started...")
    metrics_computation()
    logging.info("METRICS_COMPUTATION completed :)")
except Exception as e:
    logging.error(print(f"PROGRAM ERROR {e}"))
finally:
    logging.info("PROGRAM ENDED SUCCESFULLY :)")

