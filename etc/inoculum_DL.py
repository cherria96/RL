# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:46:18 2023

@author: Sujin
"""
#-*-coding:utf-8 -*-
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.font_manager as fm
from matplotlib import font_manager
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")


csv_path= 'final.csv'
df_raw = pd.read_csv(csv_path)
date_time = pd.to_datetime(df_raw.pop('Datetime'), format='%Y-%m-%d %H:%M:%S')

def plot_features(df,cols,time):
  plot_features = df[cols]
  plot_features.index = time
  _ = plot_features.plot(style = 'o', subplots=True, ms = 3, title = cols, legend = False, figsize = (10,10))
  plt.tight_layout()
  plt.show()
##data preprocessing##

summary = df_raw.describe().transpose()

##data visualization
df_raw = df_raw.drop(['Feeding_rate', 'Active_vol', 'Biogas','Elapsed_hours','Daily_MY', 'TOC', 'TAN', 'NH4-N'], axis = 1)
df_final = df_raw.copy()
df_final.index = date_time
##BIOGAS PREDICTION##
#1) Univariate time series prediction : ARIMA model-----------------------------------------------------------------------#
#correlation
corr_mat = df_final.corr()
biogas_corr = corr_mat['Bpr'][abs(corr_mat['Bpr'])>0.3].sort_values()
biogas_corr[:-1].plot.bar()
plt.title('Correlation betwen Bpr (p < 0.05)')
plt.show()

corr_p = pd.DataFrame(index = ['corr', 'p_value'])
for col in biogas_corr.index[:-1]:
    corr = stats.pearsonr(df_final['Bpr'],df_final[col])
    corr_p[str(col)] = [corr[0], corr[1]]   #모든 항목 p value < 0.05 --> 상관성 유의미
    
#Check stationary of biogas time series data
#1) Plotting Rolling Statistics : We find rolling mean and variance to check stationary
#2) Dickey-Fuller Test: if test statistic < critical value : time series is stationary

ts = df_final['Bpr']
# adfuller library 
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    # Dickey-Fuller test
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
    return result

# check_mean_std
def check_mean_std(ts):
    #Rolling statistics
    rolmean = ts.rolling(window = 7).mean()
    rolstd = ts.rolling(window = 7).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Elapsed time")
    plt.ylabel("Biogas production rate (L/L/d)")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()

check_mean_std(ts) #not constant mean
check_adfuller(ts) #test statistic < critical value
#We can conclude that the data is stationary

#Change data to stationary data
#1) Moving average method
window_size = 7
moving_avg = ts.rolling(window_size).mean()
plt.figure(figsize=(22,10))
plt.plot(ts, color = "red",label = "Original")
plt.plot(moving_avg, color='black', label = "moving_avg_mean")
plt.title("바이오가스생산량")
plt.xlabel("시간")
plt.ylabel("바이오가스생산량")
plt.legend()
plt.show()

ts_moving_avg_diff = ts - moving_avg
ts_moving_avg_diff.dropna(inplace=True) # first 7 is nan value due to window size
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_moving_avg_diff)
check_adfuller(ts_moving_avg_diff) #test statistic < critical values in 1% => stationary series with 99% confidence

#2) Differencing method
ts_diff = ts - ts.shift()
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("시간")
plt.ylabel("바이오가스생산량 차분")
plt.show()

ts_diff.dropna(inplace=True) # due to shifting there is nan values
# check stationary: mean, variance(std)and adfuller test
check_mean_std(ts_diff)
check_adfuller(ts_diff) #test statistic < critical values in 1% => stationary series with 99% confidence


#Time series prediction 
#ts_diff chosen for time series prediction
# ACF and PACF 
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)
lag_pacf = pacf(ts_diff, nlags=20, method='ols')

# ACF
#q = 1, cross upper confidence interval for the first time
plt.figure(figsize=(22,10))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

# PACF
#p = 1, cross upper confidence interval for the first time
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Therefore, we use (1,0,1) as parameters of ARIMA models and predict

# ARIMA LİBRARY
from statsmodels.tsa.arima.model import ARIMA
#중간에 빈 시간(2020-07 ~ 2020-11) 기점으로 데이터 분리
ts_post =ts['2022-01-28':]
# fit model
model = ARIMA(ts_post, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit = model.fit()

# predict
forecast = model_fit.predict(start='2022-11-01')

# visualization
plt.figure(figsize=(22,10))
plt.plot(ts_post.index, ts_post,label = "original")
plt.plot(forecast,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Elapsed time")
plt.ylabel("Biogas production rate")
plt.legend()
plt.show()

# predict all path
from sklearn.metrics import mean_squared_error
import math
# fit model
model2 = ARIMA(ts_post, order=(1,0,1)) # (ARMA) = (1,0,1)
model_fit2 = model2.fit()
forecast2 = model_fit2.predict()
error = mean_squared_error(ts_post, forecast2)
print("error (RMSE): " ,math.sqrt(error))
# visualization
plt.figure(figsize=(22,10))
plt.plot(ts.index, ts,label = "original")
plt.plot(forecast2,label = "predicted")
plt.title("Time Series Forecast")
plt.xlabel("Elapsed time")
plt.ylabel("Biogas production rate")
plt.legend()
plt.savefig('graph.png')
plt.show()


steps = 31
fcast = model_fit2.get_forecast(steps = steps)
fcast_sum = fcast.summary_frame(alpha = 0.10)
forecast_idx = pd.date_range('2023-01-17', periods = steps)
fcast_sum.index = forecast_idx

fig, ax = plt.subplots(figsize=(15, 5))
# Plot the data (here we are subsetting it to get a better look at the forecasts)
ts['2022-10-01':].plot(ax=ax)
# Construct the forecasts
fcast_sum['mean'].plot(ax=ax, style='k--')
ax.fill_between(fcast_sum.index, fcast_sum['mean_ci_lower'], fcast_sum['mean_ci_upper'], color='k', alpha=0.1);
plt.title('Prediction of biogas production rate (confidency: 90%)')
plt.show()

#2) Multivariate time series prediction : VAR, XGBOOST-----------------------------------------------------------------------#

#2-1)VAR
#corr value > 0.3인 항목을 input variable 
    
nobs = 6
df = df_final['2022-01-28':]
train_df = df[:-nobs]
test_df = df[-nobs:]
#Granger's Causality test
#H0: the past values of time series(X) do not cause the other series(Y)
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=30
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

#Granger's Casulaity test assumes that data is stationary      
def transform_stationary(data):
    data_diff = data - data.shift()
    plt.figure(figsize=(22,10))
    plt.plot(data_diff)
    plt.title("Differencing method") 
    plt.xlabel("Elapsed time")
    plt.ylabel(data.name + " differencing")
    plt.show()
    
    data_diff.dropna(inplace=True) # due to shifting there is nan values
    return data_diff

df_VAR = pd.DataFrame()
for col in df.columns:
    print('\n')
    ts = df_final[col]
    while True: 
        print(str(col), 'stationary check')
        result = check_adfuller(ts)
        if (result[0] < result[4]['1%']) and (result[1] < 0.05):
            df_VAR[col] = ts
            break
        else:
            ts = transform_stationary(ts)

#모두 1차 differencing 함
df_VAR = pd.DataFrame()
for col in train_df.columns:
    df_VAR[col] = transform_stationary(train_df[col])

        
GC_matrix = grangers_causation_matrix(df_VAR, variables = df_VAR.columns)        
sns.heatmap(GC_matrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
plt.title("Granger's Causality 결과")
plt.show()

df_biogas = GC_matrix.loc['Bpr_y']
df_biogas = df_biogas[df_biogas <0.051] #음식물 탈리액 pH, 습식소화조1 CH4, 가축분뇨 투입량, 습식소화조1,2 투입량
lst = df_biogas.index.to_list()
idx = []
for i in lst:
    idx.append(i.split('_x')[0])
idx.append('Bpr')

df_VAR = df_VAR[idx]


#Cointegration Test: establish the presence of statistically significant connection between two or more time series
#Order of integration: the number of differencing required to make a non-stationary time series stationary

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df_VAR) #모두 공적분 관계임을 확인함


from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(df_VAR)

#Lag order (p = 6)
aic = []
bic = []
fpe = []
hqic = []
for i in range(15):
    result = model.fit(i)
    aic.append(result.aic)
    bic.append(result.bic)
    fpe.append(result.fpe)
    hqic.append(result.hqic)
    
ax1= plt.subplot(2,2,1)
plt.plot(range(15), aic)
plt.title('AIC')

ax2 = plt.subplot(2,2,2)
plt.plot(range(15), bic, label = 'bic')
plt.title('BIC')


plt.subplot(2,2,3, sharex = ax1)
plt.plot(range(15), fpe, label = 'fpe')    
plt.title('FPE')
plt.xlabel('Lag order')

plt.subplot(2,2,4, sharex = ax2)
plt.plot(range(15), hqic, label = 'hqic')
plt.title('HQIC')
plt.xlabel('Lag order')
plt.show()

model_fitted = model.fit(6)
model_fitted.summary()

from statsmodels.stats.stattools import durbin_watson
def adjust(val, length= 6): return str(val).ljust(length)

out = durbin_watson(model_fitted.resid)
for col, val in zip(df_VAR.columns, out):
    print(adjust(col), ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 6

# Input data for forecasting
forecast_input = df_VAR.values[-lag_order:] #initial value for the forecast
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df[idx].index[-nobs:], columns=df_VAR.columns + '_1d')
df_forecast

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(train_df[idx], df_forecast, second_diff=False)        

fig, axes = plt.subplots(nrows=int(2), ncols=2, dpi=150, figsize=(20,15))
for i, (col,ax) in enumerate(zip(['TC', 'HAc', 'HPr', 'Bpr'], axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test_df[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})


print('Forecast Accuracy of: biogas production')
accuracy_prod = forecast_accuracy(df_results['Bpr_forecast'].values, test_df['Bpr'])
for k, v in accuracy_prod.items():
    print(adjust(k), ': ', round(v,4))

    
    
results = model_fitted
irf = results.irf(6)
irf.plot(orth=False, response = 'Bpr')
irf.plot_cum_effects(orth=False, response = 'Bpr')



#2) Multivariate time series prediction : XGBOOST-----------------------------------------------------------------------#
nobs = 100
df = df_final['2022-01-28':]
train_df = df[:-nobs]
test_df = df[-nobs:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

train_scaled = pd.DataFrame(train_scaled, columns = df.columns)
train_scaled.index = df.index[:-nobs]

test_scaled = pd.DataFrame(test_scaled, columns = df.columns)
test_scaled.index = df.index[-nobs:]