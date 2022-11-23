import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold,TimeSeriesSplit
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import optuna
from optuna import Trial,visualization
from optuna.samplers import TPESampler
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error,SCORERS
"""
lasso로 statistical arbitrage를 구현하여 백테스팅을 해보았다. 원래는 lasso로 index tracking을 하지만 train기간동안 excess return의 상위 5퍼센트를 구한다. 그후  원래 인덱스 수익률에 각각 더하고 빼서

overvalued와 undervalued 된 index 수익률을 계산하여 lasso로 회귀를 한다. 그런다음 beta들을 이용하여 각각 short position과 long position을 정하는 전략이다. 비록 성과는 좋지 못하였지만 구현해보는데 

의의를 두었다. 방식은 Lasso-based index tracking and statistical arbitrage long-short strategies(Leonardo Riegel Sant’Annaa, João Frois Caldeirab, Tiago Pascoal Filomenaa)를 최대한 반영하였다.


"""

start = '2010-01-04'
end = '2022-06-30'
price_daily=pd.read_csv('stock/price_real.csv',index_col=0,parse_dates=True,encoding='CP949')
price_panel=pd.read_csv('stock/price_monthly_real.csv',index_col=0,parse_dates=True,encoding='CP949')
mkt_cap=pd.read_csv('stock/kospi_cap.csv',index_col=0,parse_dates=True,encoding='CP949')
mkt_cap.replace(',','',inplace=True,regex=True)
mkt_cap=mkt_cap.astype(float)
price_daily=price_daily.loc[:'2022-07-01']
price_panel=price_panel.loc[:'2022-07-01']
rtn_daily=np.log(price_daily/price_daily.shift(1))
rtn_daily=rtn_daily.iloc[1:]
rtn_panel=np.log(price_panel/price_panel.shift(1))
rtn_panel=rtn_panel.iloc[1:]


kospi = fdr.DataReader(symbol = 'KS11', start = start, end = end)['Close']
k_rtn=np.log(kospi/kospi.shift(1))
k_rtn=k_rtn.iloc[1:]

###거래정지 데이터 추가
sus=pd.read_csv('stock/suspending2.csv',index_col=0,parse_dates=True,encoding='cp949')
sus=sus.fillna(1)

def lasso_arbitrage(rtn_daily,rtn_monthly,mkt_cap,k_rtn,sus,rf=8.9e-5,window=12,top=40,alpha=0.95):
    kf = KFold(10, shuffle=True, random_state=42)
    result=[]
    sus=sus.reindex(rtn_daily.index)

    ind=len(list(rtn_monthly.index))
    inx=list(rtn_monthly.index)
    for i in range(ind-window):

        rtn=rtn_daily.loc[str(inx[i])[:-12]:str(inx[i+window])[:-12]]
        ####상폐된 종목들이 train이나 트레이딩 기간에 들어가지 않게 최대한 걸러낸다
        stop=sus.iloc[str(inx[i])[:-12]:str(inx[i+window])[:-12]]
        acces=set(stop.loc[stop.values!=1].index)
        universe=set(rtn.daily.columns).intersection(acces)
        tickers=mkt_cap.loc[inx[i]].dropna().rank(method='first',ascending=False)[mkt_cap.loc[inx[i]].dropna().rank(method='first',ascending=False)<top].index.values
        tickers=set(tickers).intersection(universe)

        #big_rtn_daily=rtn[tickers].fillna(-0.99)
        big_rtn_daily=rtn[tickers].fillna(0)
        big_rtn_monthly=rtn_monthly[tickers]
        X_train=big_rtn_daily.iloc[:-1]
        temp=k_rtn.loc[X_train.index]-rf
        y_train_short=k_rtn.loc[X_train.index]+temp.quantile(q=alpha,interpolation='nearest')
        #print(f'y_train_short: {y_train_short}')
        
        y_train_long=k_rtn.loc[X_train.index]-temp.quantile(q=alpha,interpolation='nearest')
        #print(f'y_train_long: {y_train_long}')
        X_train=X_train.values

        param_grid = {
                    'alpha': [0,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,1.1,1.2,1.3,1.4,1.5]}


        model=Lasso()
        gs=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=multiprocessing.cpu_count(),
        cv=kf,verbose=False,scoring='neg_mean_absolute_percentage_error')
        gs.fit( X_train,y_train_long)
        best_param=gs.best_params_['alpha']
        #print(best_param)
        lasso_long=Lasso(alpha=best_param)
        lasso_long.fit(X_train,y_train_long)

        gs=GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=multiprocessing.cpu_count(),
        cv=kf,verbose=False,scoring='neg_mean_absolute_percentage_error')

        gs.fit( X_train,y_train_short)
        best_param=gs.best_params_['alpha']
        #print(best_param)
        lasso_short=Lasso(alpha=best_param)
        lasso_short.fit(X_train,y_train_short)

        lasso_short=Lasso(alpha=best_param)
        lasso_short.fit(X_train,y_train_short)

        long_coef=lasso_long.coef_/sum(lasso_long.coef_)
        short_coef=lasso_short.coef_/sum(lasso_short.coef_)
        #long_short=long_coef-short_coef
        #print(f'long: {big_rtn_monthly.loc[inx[i]]}\n long_coef: {long_coef} \n short_coef: {short_coef}')

        long_rtn=np.dot(long_coef,big_rtn_monthly.loc[inx[i]].fillna(0))
        short_rtn=np.dot(short_coef,big_rtn_monthly.loc[inx[i]].fillna(0))
        p_rtn=long_rtn-short_rtn
        result.append(p_rtn)

    result=np.array(result)
    portfolio=pd.DataFrame(result,index=list(rtn_monthly.index)[window:])
    portfolio.rename(columns={0 : "returns"},inplace=True)

    return portfolio

# def lasso_arbitrage(rtn_daily,rtn_monthly,mkt_cap,k_rtn,top=51,alpha=0.025):
#     kf = KFold(10, shuffle=True, random_state=42)
#     result=[]

#     ind=len(list(rtn_monthly.index))
#     inx=list(rtn_monthly.index)
#     for i in range(ind-3):

#         rtn=rtn_daily.loc[str(inx[i])[:-12]:str(inx[i])[:-12]]
#         tickers=mkt_cap.loc[inx[i]].dropna().rank(method='first',ascending=False)[mkt_cap.loc[inx[i]].dropna().rank(method='first',ascending=False)<11].index.values
#         tickers=list(tickers)
#         big_rtn_daily=rtn[tickers].fillna(-0.99)
#         big_rtn_monthly=rtn_monthly[tickers]
#         X_train=big_rtn_daily.iloc[:-1]
#         y_train_long=k_rtn.loc[X_train.index].values+alpha
#         X_train=X_train.values
#         y_train_short=y_train_long-2*alpha


#         def objectiveLasso_long(trial:Trial) -> float:
#             score_list=[]
#             params={'alpha':trial.suggest_float('alpha', 0,0.001)}
#             for tr_idx, val_idx in kf.split(X_train, y_train_long):
#                 X_tr, X_val = np.array(X_train)[tr_idx], np.array(X_train)[val_idx]
#                 y_tr, y_val = np.array(y_train_long)[tr_idx], np.array(y_train_long)[val_idx]
#                 lasso = Lasso(**params)
#                 lasso.fit(X_tr, y_tr)
#                 predictions = lasso.predict(X_val)
#                 predictions = np.expm1(predictions)
#                 rmse = np.sqrt(mean_squared_error(np.expm1(y_val), predictions))
#                 score_list.append(rmse)
#             return np.mean(score_list)
#         lasso_study_long=optuna.create_study(
#                 study_name = 'lasso_test',
#                 direction='minimize',
#                 sampler = TPESampler(seed=42)
#             )
#         lasso_study_long.optimize(objectiveLasso_long,n_trials=20)

#         best_param=lasso_study_long.best_trial.params
#         best_param=float(best_param['alpha'])

#         lasso_long=Lasso(alpha=best_param)
#         lasso_long.fit(X_train,y_train_long)




#         def objectiveLasso_short(trial:Trial) -> float:
#             score_list=[]
#             params={'alpha':trial.suggest_float('alpha',  0,0.001)
#             }
#             for tr_idx, val_idx in kf.split(X_train, y_train_short):
#                 X_tr, X_val = np.array(X_train)[tr_idx], np.array(X_train)[val_idx]
#                 y_tr, y_val = np.array(y_train_short)[tr_idx], np.array(y_train_short)[val_idx]
#                 lasso = Lasso(**params)
#                 lasso.fit(X_tr, y_tr)
#                 predictions = lasso.predict(X_val)
#                 predictions = np.expm1(predictions)
#                 rmse = np.sqrt(mean_squared_error(np.expm1(y_val), predictions))
#                 score_list.append(rmse)
#             return np.mean(score_list)
#         lasso_study_short=optuna.create_study(
#                 study_name = 'lasso_test',
#                 direction='minimize',
#                 sampler = TPESampler(seed=42)
#             )
#         lasso_study_short.optimize(objectiveLasso_short,n_trials=20)
#         best_param=lasso_study_short.best_trial.params
#         best_param=float(best_param['alpha'])

#         lasso_short=Lasso(alpha=best_param)
#         lasso_short.fit(X_train,y_train_short)

#         long_coef=lasso_long.coef_/sum(lasso_long.coef_)
#         short_coef=lasso_short.coef_/sum(lasso_short.coef_)
#         #long_short=long_coef-short_coef
#         long_rtn=np.dot(long_coef,big_rtn_monthly.loc[inx[i]])
#         short_rtn=np.dot(short_coef,big_rtn_monthly.loc[inx[i]])
#         p_rtn=long_rtn-short_rtn
#         result.append(p_rtn)

#     result=np.array(result)
#     portfolio=pd.DataFrame(result,index=list(rtn_monthly.index)[3:])
#     portfolio.rename(columns={0 : "returns"},inplace=True)

#     return portfolio
result=lasso_arbitrage(rtn_daily=rtn_daily,rtn_monthly=rtn_panel,k_rtn=k_rtn,mkt_cap=mkt_cap)
print(result)