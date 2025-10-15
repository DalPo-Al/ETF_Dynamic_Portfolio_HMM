import yfinance as yf
import os
import pandas as pd

def fetch_data(ticker, start, end, save_path="data/data.feather", interval="1d"):
  #if file exist we do not download again
  if os.path.exists(save_path):
    df=pd.read_feather(save_path).set_index("Date")
  else:
    df=yf.download(tickers=ticker, 
                     start=start,
                     end=end,
                     interval=interval)["Close"]
    #filna using forward fill method (reuse the previous available information)
    df.ffill(inplace=True)
    #drop any remained Nan value (safety net before computing feature engineering)
    df.dropna(inplace=True, how="any")
    #we reset the index before saving the dataframe to feather, feather wants integers as index
    df.reset_index(inplace=True)
    #save to feather for usage
    df.to_feather(save_path)
    #save to csv for visualization
    df.to_csv("data/data.csv")
    #set index to Date
    df.set_index("Date")

  return df

if __name__=="__main__":
  tickers = ["SPY", "BND", "GLD", "USO", "IEF"]
  start_date = "2013-01-01"
  end_date = "2023-09-01"
  print(fetch_data(ticker=tickers, start=start_date,end=end_date).head())
