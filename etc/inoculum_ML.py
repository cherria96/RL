# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 17:05:39 2022

@author: Sujin
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
'''
scope = [
'https://spreadsheets.google.com/feeds',
'https://www.googleapis.com/auth/drive',
]

json_file_name = 'inoculum-var-4b2cc760664a.json'
credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
gc = gspread.authorize(credentials)
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1oqE93Vwt5F-_zIBCXp0aJtduTr0wcMtXIC9azUKb_lo/edit#gid=498946974'

# 스프레스시트 문서 가져오기 
doc = gc.open_by_url(spreadsheet_url)

# 시트 선택하기
worksheet = doc.worksheet('Summary')
data = pd.DataFrame(worksheet.get_all_records())
'''
#구글 시트 API error --> just downloaded csv
data = pd.read_csv('Dat_25L AnR_python - Summary.csv')
data.columns = data.iloc[2,:].values
data = data.drop([0,1,2,3,4])
data = data.drop(data.columns[17], axis = 1)
data['Datetime'] = data['Date'] + ' ' + data['Time']
data['Datetime'] = pd.to_datetime(data['Datetime'])

R1 = data.iloc[:,7:38]
R1 = R1.drop('specific OLR (gVS/L/d)', axis = 1)
R1 = pd.concat([R1, data.iloc[:,2]], axis = 1)
R1.index = data['Datetime']

columns = ['specific_OLR', 'Th_working_vol','Feeding_vol', 'Feeding_rate','Sampling_vol','Active_vol', 'pH', 'ORP', 
           'Alkalinity', 'PA/IA', 'TS', 'VS', 'VSS', 'COD', 'sCOD', 'TOC', 'TC',
           'Protein','TAN', 'NH4-N', 'NH3-N', 'Lipid','TVFA', 'HAc', 'HPr', 'Biogas', 'Bpr','stdCH4', 'stdCO2', 'Daily_MY', 'Elapsed_hours']

R1.columns = columns

use_cols = ['specific_OLR', 'Feeding_rate', 'Active_vol', 'pH', 'ORP', 'Alkalinity', 'TS', 'VS', 'VSS', 'COD', 'sCOD', 'TOC', 'TC', 'Protein','Lipid',
           'TAN', 'NH4-N', 'TVFA', 'HAc', 'HPr', 'Biogas', 'Bpr', 'stdCH4', 'stdCO2', 'Daily_MY', 'Elapsed_hours']

use_R1 = R1[use_cols]
#---------------------------------------------------------------------------------------------------#
enddate = '2023-01-09'
df = use_R1[ : enddate].copy()

df[use_cols] = df[use_cols].replace({0:np.nan})
idx = df[df.TVFA.isnull()].index
df.loc[idx, 'HPr'] = np.nan
df = df.replace({'#REF!':np.nan})
df = df.replace({'#N/A':np.nan})
df = df.replace({'':np.nan})
df = df.replace({'#DIV/0!':np.nan})
df = df.astype('float64')
df['Elapsed_hours'].fillna(0, inplace = True)
#df = df.drop('2022-05-22 00:00:00')



#Effluent characteristics visualization--------------------------------------------------------------#

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize = (15,20))
line1 = ax1.scatter(df.index, df.Feeding_rate, label = 'Feeding rate')
ax12 = ax1.twinx()
line2 = ax12.scatter(df.index, df.specific_OLR, label = 'specific OLR', color = 'red' )
ax1.legend(handles = [line1, line2])

line1 = ax2.scatter(df.index, df.pH, label = 'pH')
ax22 = ax2.twinx()
line2 = ax22.scatter(df.index, df.TVFA, label = 'TVFA', color = 'red')
ax2.legend(handles = [line1, line2])

ax3.scatter(df.index, df.VS, label = 'VS')
ax3.scatter(df.index, df.COD, label = 'COD',color = 'red')
ax3.legend()

line1 = ax4.scatter(df.index, df.Bpr, label = 'Biogas production rate')
ax42 = ax4.twinx()
line2 = ax42.scatter(df.index, df.Daily_MY, label = 'methane yield', color = 'red')
ax4.legend(handles = [line1, line2])

line1 = ax5.scatter(df.index, df.Feeding_rate, label = 'Feeding rate')
ax52 = ax5.twinx()
line2 = ax52.scatter(df.index, df.Bpr, label = 'Biogas production rate', color = 'red')
ax5.legend(handles = [line1, line2])

#plt.savefig('effluent characteristics.eps', format = 'eps')
plt.show()

#Influent feeding visualization------------------------------------------------------------------------#

FR = df.groupby(by = [df.index.month])['Feeding_rate'].mean()
OLR = df.groupby(by = [df.index.month])['specific_OLR'].mean()

fig, ax1 = plt.subplots(figsize = (8,8))
colors = sns.color_palette('summer', len(FR))
bar1 = ax1.bar(FR.index, FR.values, label = 'Feeding rate', color = colors)
ax1.set_ylabel('Feeding rate (L/d)')
ax1.set_xlabel('month')
ax2 = ax1.twinx()
line1, = ax2.plot(OLR.index, OLR.values, color = 'b', linestyle = '--', marker = 'o', label = 'specific OLR')
ax2.set_ylabel('specific OLR (gCOD/L/day)')
ax1.legend(handles = [bar1, line1], loc= 'lower right')

plt.title('Substrate feeding monthly average')
plt.show()

#interpolation of missing values-----------------------------------------------------------------------#
#fillna for endpoint(230109) with the last data
endpoint = df.iloc[-1,:]
lastdata = []
for col in df.columns :
    if endpoint.isna()[col]:
        data = df[col].dropna()[-1]
        lastdata.append(data)
    
    else:
        lastdata.append(endpoint[col])
df.iloc[-1,:] = lastdata
        

def monotonInterpolate(data, item):
    df_interpolate  = data[['Elapsed_hours', item]].copy()
    df_interpolate = df_interpolate.dropna(subset = item)
    f_interpolate = interpolate.PchipInterpolator(df_interpolate['Elapsed_hours'], df_interpolate[item])
    spline_df = f_interpolate(data['Elapsed_hours'])
    return df_interpolate, spline_df

def interpolateplot(columns, axs, interpolated):
    for item,ax in zip(columns, axs.flat):
        df_interpolate, spline_df = monotonInterpolate(interpolated, item)
        ax.plot(df_interpolate.index, df_interpolate[item], 'ko', interpolated.index, spline_df, 'r--')
        ax.set_ylabel('{}'.format(item))
        interpolated[item] = spline_df
    fig.legend(['real value', 'monotonic spline'])
    plt.suptitle('monotonic spline interpolation')
    fig.tight_layout()
    plt.show()
    return interpolated

columns1 = ['pH', 'ORP', 'Alkalinity','TS', 'VS', 'VSS']
columns2 = ['COD', 'TC','sCOD', 'TOC',  'TVFA', 'HPr', 'HAc']
columns3 = ['Biogas', 'Bpr', 'stdCH4', 'stdCO2']

interpolated = df.copy()

fig, axs = plt.subplots(3,2, figsize = (20,15), sharex = True)
interpolated = interpolateplot(columns1, axs, interpolated)

fig, axs = plt.subplots(4,2, figsize = (20,15), sharex = True)
interpolated = interpolateplot(columns2, axs, interpolated)

fig, axs = plt.subplots(2,2, figsize = (20,12), sharex = True)
interpolated = interpolateplot(columns3, axs, interpolated)

interpolated.to_csv('final.csv')

#interpolated = interpolated.fillna(0)

#Outlier detection ---------------------------------------------------------------------------------------#
'''
df_delna = df.copy()
col = 'COD'
df_delna = df_delna.dropna(subset = col)

#STL decomposition Anomaly Detection ----------------------------------------------------------------------------------------#
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df_delna[col],model='additive', period = 7)
plt.rc('figure',figsize=(12,8))
plt.rc('font',size=15)
fig = result.plot()

fig, (ax1,ax2) = plt.subplots(2, sharex = True)
argmax = np.argmax(df_delna[col])
argmin = np.argmin(df_delna[col])
x = result.resid.index
y = result.resid.values
ax1.plot_date(x, df_delna[col], color = 'black', linestyle = '--')
ax1.plot(x[argmax], df_delna[col][argmax], 'ro', markersize = 15, alpha = 0.5)
ax1.plot(x[argmin], df_delna[col][argmin], 'ro', markersize = 15, alpha = 0.5)
ax1.set_title('{} value'.format(col))
ax2.plot_date(x, y, color = 'black')
ax2.annotate('Anomaly', (mdates.date2num(x[argmax]), y[argmax]), xytext=(30, 20), 
           textcoords='offset points', color='red',arrowprops=dict(facecolor='red',arrowstyle='fancy'))
ax2.axhline(2, color = 'red', linestyle = '--', alpha = 0.4)
ax2.axhline(0, color = 'black', linestyle = '--', alpha = 0.4)
ax2.axhline(-2, color = 'red', linestyle = '--', alpha = 0.4)
ax2.set_title('Residue result of {} value'.format(col))
fig.tight_layout()
plt.show()

#CART (Classification and Regression Tree)------------------------------------------------------------------#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
outliers_fraction = float(.01)
scaler = StandardScaler()
np_scaled = scaler.fit_transform(df_delna[col].values.reshape(-1, 1))
data = pd.DataFrame(np_scaled)
model =  IsolationForest(contamination=outliers_fraction)
model.fit(data) 
df_delna['anomaly'] = model.predict(data)

fig, ax = plt.subplots(figsize=(12,8))
a = df_delna.loc[df_delna['anomaly'] == -1, [col]] #anomaly
ax.plot_date(df_delna[col].index, df_delna[col], color='black', linestyle = '--', label = 'Normal')
ax.plot(a.index,a[col],'ro', label = 'Anomaly')
plt.legend()
plt.title('Outlier detection with CART')
plt.show()


#Clustering based anomaly detection ---------------------------------------------------------------------#

from sklearn.cluster import KMeans
data = df[['TS', 'COD', 'specific_OLR', 'Biogas', 'pH', 'sCOD']]
data.dropna(inplace = True)
n_cluster = range(1,20)
kmeans = [KMeans(n_clusters = i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(n_cluster, scores, color = 'darkblue')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.grid(True)
plt.show()

km = KMeans(n_clusters = 8)
km.fit(data)
km.predict(data)
labels = km.labels_

fig, ax = plt.subplots(figsize = (10,6))
ax = Axes3D(fig, rect = [0,0,0.95,1], elev =35, azim = 120,auto_add_to_figure = False)
fig.add_axes(ax)
ax.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c = labels.astype(np.float64), edgecolor = 'k')
ax.set_xlabel('TS')
ax.set_ylabel('COD')
ax.set_zlabel('specific_OLR')
plt.title('K Means', fontsize = 14)

X = data.values
X_std = StandardScaler().fit_transform(X)
#Calculating Eigenvecors and eigenvalues of Covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key = lambda x: x[0], reverse= True)
# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3, align='center', label='individual explained variance', color = 'y')
plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

data_std = pd.DataFrame(X_std)
pca = PCA(n_components=3)
data_std = pca.fit_transform(data_std)
scaler = StandardScaler()
np_scaled = scaler.fit_transform(data_std)
data_std = pd.DataFrame(np_scaled)

kmeans = [KMeans(n_clusters=i).fit(data_std) for i in n_cluster]
data['cluster'] = kmeans[7].predict(data_std)
data.index = data_std.index
data['principal_feature1'] = data_std[0]
data['principal_feature2'] = data_std[1]
data['principal_feature3'] = data_std[2]
data['cluster'].value_counts()

def getDistanceByPoint(data_std, model):
   distance = pd.Series(dtype = 'float64')
   for i in range(0,len(data_std)):
       Xa = np.array(data_std.loc[i])
       Xb = model.cluster_centers_[model.labels_[i]-1]
       distance.at[i]=np.linalg.norm(Xa-Xb)
   return distance
outliers_fraction = 0.1
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(data_std, kmeans[7])
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly)
data['anomaly1'] = (distance >= threshold).astype(int)

fig, ax = plt.subplots(figsize=(10,6))
colors = {0:'blue', 1:'red'}
ax.scatter(data['principal_feature1'], data['principal_feature2'], c=data["anomaly1"].apply(lambda x: colors[x]))
plt.xlabel('principal feature1')
plt.ylabel('principal feature2')
plt.show();

tmp = df[['TS', 'COD', 'specific_OLR', 'Biogas', 'pH', 'sCOD']]
tmp.dropna(inplace = True)
data['date_time'] = tmp.index
data = data.sort_values('date_time')

lst = ['TS', 'COD', 'specific_OLR', 'Biogas', 'pH', 'sCOD']
fig, axes= plt.subplots(3,2, figsize=(15,10))
for col, ax in zip(lst, axes.flat):    
    a = data.loc[data['anomaly1'] == 1, ['date_time', col]] #anomaly
    ax.plot_date(data['date_time'], data[col], color='k',label='Normal', linestyle = '--')
    ax.plot(a['date_time'],a[col], 'ro',label='Anomaly')
    ax.xaxis_date()
    ax.set_xlabel('Date Time')
    ax.set_ylabel(col)
fig.autofmt_xdate()
plt.suptitle('Outlier detection with Kmeans clustering', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.tight_layout()
plt.show()

#forecasting -----------------------------------------------------------------------------------------------#
from prophet import Prophet

def t_df(data, col):
    t = pd.DataFrame()
    t['ds'] = data.index
    t['y'] = data[col].values
    return t

def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.8):
    m = Prophet(daily_seasonality = False, yearly_seasonality = False, weekly_seasonality = False,
                seasonality_mode = 'additive', 
                interval_width = interval_width,
                changepoint_range = changepoint_range)
    m = m.fit(dataframe)
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)
    return forecast
   
def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    #anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
    
    return forecasted

def plot_anomalies(forecasted, ax, a):
    ax.plot_date(forecasted['ds'], forecasted['fact'], color = 'black', label = 'normal')
    ax.scatter(a.ds, a.fact, c = 'red', s = abs(a.importance/a.importance.mean())*300, label = 'anomaly', alpha = 0.5)   
    ax.fill_between(forecasted['ds'], forecasted['yhat_lower'], forecasted['yhat_upper'], alpha = 0.2)


def multiple_plots(columns, axs, df):
    df_out = df.copy()
    for col, ax in zip(columns, axs.flat):
        t = t_df(df, col)
        forecast = fit_predict_model(t)
        pred = detect_anomalies(forecast)
        a = pred.loc[(pred['anomaly'] == 1) |(pred['anomaly'] ==-1)]
        plot_anomalies(pred, ax, a)
        ax.set_ylabel('{}'.format(col))    
        for i in a.ds:
            idx = df_out[df_out.index == i].index
            if idx > '2022-02-01':              
                df_out.loc[idx, col] = np.nan   #remove outlier in steady state
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.suptitle('Outlier detection with Time series forecasting', fontsize = 20)
    fig.tight_layout()
    plt.show()
    return df_out

columns1 = ['pH', 'ORP', 'Alkalinity','TS', 'VS', 'VSS', 'COD', 'TC']
columns2 = ['sCOD', 'TAN', 'NH4-N', 'TVFA', 'HAc', 'HPr'] 
columns3 = ['Biogas', 'Bpr', 'stdCH4', 'stdCO2']

fig, axs = plt.subplots(4,2, figsize = (20,15), sharex = True)
df_out1 = multiple_plots(columns1, axs, df)

fig, axs = plt.subplots(3,2, figsize = (20,12), sharex = True)
df_out2 = multiple_plots(columns2, axs, df_out1)

fig, axs = plt.subplots(2,2, figsize = (20,10), sharex = True)
df_out3 = multiple_plots(columns3, axs, df_out2)
'''
#interpolation with outlier removed (outlier detected from time forecasting) -----------------------------------------------------------------------------------------#
interpolated = df_out3.copy()
interpolated.iloc[:2,:].fillna(0, inplace = True)
interpolated.iloc[1, 4] = np.nan #ORP replace 0 with NA

fig, axs = plt.subplots(4,2, figsize = (20,15), sharex = True)
interpolated = interpolateplot(columns1, axs, interpolated, )

fig, axs = plt.subplots(3,2, figsize = (20,12), sharex = True)
interpolated = interpolateplot(columns2, axs, interpolated)

fig, axs = plt.subplots(2,2, figsize = (20,10), sharex = True)
interpolated = interpolateplot(columns3, axs, interpolated)

final= interpolated.drop(['TOC', 'Elapsed_hours'], axis =1)

#feeding volume prediction -------------------------------------------------------------------------------------------------------#
nafinal = interpolated.copy()



