import numpy as np
import pandas as pd

def feature_engineering(momentum_window, volatility_window):
  df=pd.read_feather("data/data.feather").set_index("Date")
  #return computation
  returns=np.log(df/df.shift(1))
  returns.columns=[f"ret_{cls}" for cls in returns.columns]
  df=pd.concat([df, returns], axis=1)
  #momentum computation
  momentum=returns.rolling(momentum_window).sum()
  momentum.columns=[f"mom_{cls}" for cls in momentum.columns]
  df=pd.concat([df, momentum], axis=1)
  #volatility computation
  volatility=returns.rolling(volatility_window).std()
  volatility.columns=[f"vol_{cls}"for cls in volatility.columns]
  df=pd.concat([df, volatility], axis=1)
  #remove NaN values
  df.dropna(inplace=True, how="any")
  #data saved
  df.to_feather("data/data.feather")
  df.to_csv("data/data.csv")
  return df  

if __name__=="__main__":
  df=pd.read_feather("data/data.feather").set_index("Date")
  print(feature_engineering(momentum_window=60, volatility_window=20).head(70))
