# Here we define the covariance matrix and the mean value using the rolling window approach simulating a real life forward trained portfolio algo
import pandas as pd
import numpy as np
import joblib
import cvxpy as cp
import matplotlib.pyplot as plt

def comp_mean_covariance():
  data=pd.read_feather("data/hidden_state.feather")
  window=252
  output={}
  for i in range(0, data.shape[0]-window+1):
    data_window=data.iloc[i:i+window, :] #slice of dataframe
    key_window=data_window.index[-1] #date of the head of slice IMPORTANT
    output[key_window]={}
    for g in data_window["Hidden_state_t+1"].unique():
      condition=(data_window["Hidden_state_t+1"]==g) #to filter per hidden state
      sub=data_window.where(condition).loc[:,"ret_BND":"ret_USO"]
      sub_covariance=sub.cov().values
      sub_mean=sub.mean().values
      output[key_window][g]={
        "mean":sub_mean,
        "covariance_matrix":sub_covariance
      }     
  joblib.dump(output, "output_dict.joblib")
  output=joblib.load("output_dict.joblib") #save as joblib object to avoid recompilation during developement, then we simply return the output

#NOTE
#The hidden_state feather file contains at time t the hidden state for t+1 estimated using HMM statistical model,
#we are interested in fitting a MVO portfolio estimation for each hidden state predicted at time t for next day, knowing 
#so which weights to use to define the portfolio for next period.

#NOTE
#we are interested in taking the signal from HMM model and convert it into weights in order to build a portfolio. 
#We are interested in comparing three portfolio optimization methods:
#- Mean Variance Optimization (MVO)
#- Global Minimum Variance (GMV)
#- Equal weights (EW)
#Ideally the portfolio optimization 1 and 2 should outperform the benchmark 3 since we are using the signals from HMM as input.

def mvo_optimization(mu, sigma):
  n=len(mu)
  w=cp.Variable(n)
  gamma=3.0 #risk aversion
  risk=cp.quad_form(w, sigma)
  ret=w@mu
  objective=cp.Maximize(ret-(gamma/2)*(risk))
  constraints=[
    cp.sum(w)==1, #total investment constraint
    w>=0,         #NO short selling
    w<=0.3        #ensure diversification
    ]
  prob=cp.Problem(objective=objective, constraints=constraints)
  prob.solve()
  return np.round(w.value, 3)

def gmv_optimization(mu, sigma):
  w=cp.Variable(len(mu))
  risk=cp.quad_form(w, sigma)
  objective=cp.Minimize(risk)
  constraints=[
    cp.sum(w)==1, #total investment constraint
    w>=0,         #NO short selling
    w<=0.3        #ensure diversification
  ]
  prob=cp.Problem(constraints=constraints, objective=objective)
  prob.solve()
  return np.round(w.value, 3)

def ew_optimization(mu):
  n=len(mu)
  const=1/n
  w=[const for _ in range(n)]
  return np.round(w, 3)

def optimization():
  data=pd.read_feather("data/hidden_state.feather")
  #print(data.head())
  #print(data.index[-1])
  dictionary=joblib.load("output_dict.joblib")
  #print(list(dictionary.keys())[-1])
#
  #print(data.index[0])
  #print(list(dictionary.keys())[0])
  dictionary_dates=list(dictionary.keys())

  for d in dictionary_dates:
    hidden_pred=data.loc[d, "Hidden_state_t+1"]
    mean=dictionary[d][hidden_pred]["mean"]
    covariance_matrix=dictionary[d][hidden_pred]["covariance_matrix"]
    #print(f"problem iteration {d}\n")
    #print(f"mean {mean}")
    #print(f"covariance_matrix {covariance_matrix}")
    w_mvo=mvo_optimization(mu=mean, sigma=covariance_matrix)
    #print(f"optimal_weights_MVO {w_mvo}")
    w_gmv=gmv_optimization(mu=mean, sigma=covariance_matrix)
    #print(f"optimal_weights_GMV {w_gmv}")
    w_ew=ew_optimization(mu=mean)
    #print(f"optimal_weights_EW {w_ew}\n")
    #add to dictionary in order to have access to weights
    dictionary[d]["w_MVO"]=w_mvo
    dictionary[d]["w_GMV"]=w_gmv
    dictionary[d]["w_EW"]=w_ew
  joblib.dump(dictionary, "output_dict.joblib")
  dictionary=joblib.load("output_dict.joblib")

def returns_computation():
  dictionary=joblib.load("output_dict.joblib")
  data=pd.read_feather("data/hidden_state.feather")
  dictionary_keys=list(dictionary.keys())  
  
#NOTE
#The problem here is that the return is computed using t and t-1 while weights are predicted for t+1, 
# this is a problem that we fix using two indeces
  t_0=dictionary_keys[2:-1] #iterator for current date            => used for return
  t_1=dictionary_keys[1:-2] #iterator for day before date         => used for current weights 
  t_2=dictionary_keys[0:-3] #iterator for day before before date  => used for previous weights
  
  #print(f"t-0: {t_0[-1]}")
  #print(f"t-1: {t_1[-1]}")
  #print(f"t-2: {t_2[-1]}")
  
  port_ret=[[],[],[]] #MVO, GMV, EW
  cost=[[],[],[]] #MVO, GMV, EW
  for t0, t1, t2  in zip(t_0, t_1, t_2):
    returns=data.loc[t0, "ret_BND":"ret_USO"]
    #current weights
    w_mvo=dictionary[t1]["w_MVO"]
    w_gmv=dictionary[t1]["w_GMV"]
    w_ew=dictionary[t1]["w_EW"]

    #previous weights
    w_mvo_prev=dictionary[t2]["w_MVO"]
    w_gmv_prev=dictionary[t2]["w_GMV"]
    w_ew_prev=dictionary[t2]["w_EW"]

    #turnover in weights
    cost_factor=0.05/100 #turnover cost 0.05% of per unit turnover
    cost_mvo=np.round(np.sum(np.abs(w_mvo-w_mvo_prev)), 3)*cost_factor
    cost_gmv=np.round(np.sum(np.abs(w_gmv-w_gmv_prev)), 3)*cost_factor
    cost_ew=np.round(np.sum(np.abs(w_ew-w_ew_prev  )), 3)*cost_factor

    cost[0].append(cost_mvo)
    cost[1].append(cost_gmv)
    cost[2].append(cost_ew)

    #cost accounting on returns
    port_ret[0].append(np.round(np.dot(returns-cost_mvo, w_mvo), 3)) #compute portfolio return
    port_ret[1].append(np.round(np.dot(returns-cost_gmv, w_gmv), 3))
    port_ret[2].append(np.round(np.dot(returns-cost_ew, w_ew ), 3))


  df=pd.DataFrame(
    {
      "ret_mvo":port_ret[0],
      "ret_gmv":port_ret[1],
      "ret_ew":port_ret[2],
      "cost_turnover_mvo":cost[0],
      "cost_turnover_gmv":cost[1],
      "cost_turnover_ew":cost[2]
    }, index=t_0
    )
  
  df.to_feather("data/output.feather")
  df.to_csv("data/output.csv")
  #print(df)
  cum_return=(1+df.loc[:,["ret_mvo", "ret_gmv", "ret_ew"]]).cumprod()-1
  return cum_return.iloc[-1]

def sharpe_ratio_computation(annual_risk_free, returns):
  daily_risk_free=((1+annual_risk_free)**(1/252))-1
  sharpe_ratio=(returns.mean()-daily_risk_free)/returns.std()
  return sharpe_ratio*np.sqrt(252) #annualized sharpe ratio  

def max_draw_down_computation(returns):
  cumulative=(1+returns).cumprod()-1 #daily returns
  peak=cumulative.cummax()
  drawdown=(cumulative-peak)/peak #drawdown series
  max_drawdown=drawdown.min()
  return max_drawdown

def metrics_computation():
  data=pd.read_feather("data/output.feather")
  df=pd.DataFrame()
  risk_free=0.043 #1 year return of US TBILL
  returns=data.loc[:,["ret_mvo", "ret_gmv", "ret_ew"]]
  
  #SHARPE RATIO
  sharpe_ratio=sharpe_ratio_computation(annual_risk_free= risk_free, returns=returns)
  df["Sharpe_Ratio"]=np.round(sharpe_ratio, 5)

  #MAX DRAW DOWN
  max_drawdown=max_draw_down_computation(returns= returns)
  df["Max_Draw_Down"]=np.round(max_drawdown, 3)

  #FINAL COMPOUNDED RETURN
  comp_return=returns_computation()
  df["Comp_return_final [%]"]=np.round(comp_return*100, 2)
  
  df.to_csv("data/metrics.csv")
  print(df)

def plotting():
  data=pd.read_feather("data/output.feather")
  returns=data.loc[:,["ret_mvo", "ret_gmv", "ret_ew"]]
  cost=data.loc[:,["cost_turnover_mvo","cost_turnover_gmv","cost_turnover_ew"]]
  cumulative=(1+returns).cumprod()-1  #cumulative return of portfolio
  cumulative_cost=cost.cumsum()       #cumulative sum of turnover cost
  fig,ax=plt.subplots(2,1, sharex=True, figsize=(12, 6))
  ax[0].plot(cumulative, label=["MVO", "GMV","EW"])
  ax[0].set_title("Cumulative_return_net [log10]")
  ax[0].legend()
  ax[0].grid(True)
  ax[1].plot(np.log10(cumulative_cost), label=["MVO", "GMV","EW"]) #log(1+x)
  ax[1].set_title("Turnover_cost") #when flat the cost is the same.
  ax[1].legend()
  ax[1].grid(True)
  plt.savefig("data/plot.jpg", dpi=300)
  plt.show()
    
if __name__=="__main__":
  #comp_mean_covariance() #RUN COVARIANCE - MEAN COMPUTATION FOR HMM SIGNAL USING WINDOW
  #optimization()         #RUN PORTFOLIO OPTIMIZATION AND SAVE WEIGHTS INTO SAME DICTIONARY USED and save it as joblib object
  #returns_computation()  #RUN PORTFOLIO RETURNS COMPUTATION AND SAVE TO data/output.feather file
  plotting()             #RUN PLOTTING
  #metrics_computation()  #RUN METRICS COMPUTATION

