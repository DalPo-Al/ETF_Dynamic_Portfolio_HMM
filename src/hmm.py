import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA #PCA decomposition
from hmmlearn.hmm import GaussianHMM #HMM model
import os #to have access to os paths

def bic_computation(model, X):
  log_likelihood_function=model.score(X)
  n_params = (
    model.n_components * X.shape[1] * (X.shape[1] + 1) / 2  # covariances
    + model.n_components * X.shape[1]                      # means
    + model.n_components * (model.n_components - 1)       # transitions
    + model.n_components - 1                               # initial probs
    )
  bic=-2*log_likelihood_function+n_params*np.log(X.shape[0])
  return np.round(bic, 2)

def fit_pca(random_state, window_data):
  transformer=PCA(n_components=0.9, svd_solver="full", random_state=random_state)
  pca=transformer.fit_transform(window_data)
  return pca

def optimal_components(candidate, window):
  candidate_hidden_state=[i+1 for i in range(candidate)]
  bic_list=[]
  for i in candidate_hidden_state:
    hmm=GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000).fit(window)
    bic_list.append(bic_computation(model=hmm, X=window))
  df=pd.DataFrame(
    bic_list,
    index=candidate_hidden_state,
    columns=["BIC_value"]
  )
  return df["BIC_value"].idxmin() #return index of min value in BIC column

def fit_hmm(window, random_state, n_components):
  hmm=GaussianHMM(random_state=random_state, n_components=n_components, covariance_type="diag", n_iter=2000).fit(window)
  #predict hidden state for every observation of window
  hidden_state=hmm.predict(window)
  #take last hidden state as current state
  transition_matrix=hmm.transmat_[hidden_state[-1]]
  #return prediction for next hidden state
  prediction=transition_matrix.argmax()
  #print(f"Trantition matrix: {transition_matrix}")
  #print(f"Next hidden state: {transition_matrix.argmax()}")
  return prediction

def hmm_model(window=252):
  #if hidden state extraction has been already done, do not repeat it.
  if os.path.exists(path="data/hidden_state.feather"):
    df=pd.read_feather("data/hidden_state.feather")
    print("COMPUTATION ALREADY DONE, check data/hidden_state.feather")
  #else compute hidden state extraction
  else:
    n_components=3
    random_state=42
    data=pd.read_feather("data/data.feather")
    in_window=data.iloc[0:window,8:17] #momentum and volatility only
    hidden_state_prediction_list=[]
    for i in range(data.shape[0]-window+1):
      start=i
      end=i+window
      in_window=data.iloc[start:end,8:17] #<= 8 to 17 are the indexes for momentum and volatility
      pca_in_window=fit_pca(random_state=random_state, window_data=in_window)
      #check if Nan are present in pca components obtained
      #GaussianHMM model cannot work with Nan, the model would break down
      if np.isnan(pca_in_window).any():
        #if true, replace Nan with zero, +inf with large finite number and -inf with large negative number
        pca_in_window=np.nan_to_num(pca_in_window) 

      hidden_state_prediction=fit_hmm(window=pca_in_window, random_state=random_state, n_components=n_components)
      hidden_state_prediction_list.append(hidden_state_prediction)
    df=pd.DataFrame(
      hidden_state_prediction_list,
      index=data.index[window-1:],
      columns=["Hidden_state_t+1"]
    )
    #print(df.head())
    #print(df.tail())
    prediction_df=data.iloc[window-1:].join(df)
      
    #SAVE
    prediction_df.to_csv("data/hidden_state.csv")
    prediction_df.to_feather("data/hidden_state.feather")
    #we return nothing since we work with feather file

if __name__=="__main__":
  hmm_model()