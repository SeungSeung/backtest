import pandas as pd
from sqlalchemy import create_engine
import pymysql
import numpy as np
import plotly.express as px



def mean_stock_movement(signal_num,data,signal_df, end_date,minus_end_date=0,mean_only=False):
    backtest_data=data.copy()
    ###외부상장 종목 제거
    backtest_data=backtest_data.loc[(backtest_data['market']!='외감') & (backtest_data['market']!='KONEX')]
    ###거래정지 종목 제거
    backtest_data=backtest_data.loc[backtest_data['trading_suspension']!=1]
    ##상장종목 0인거 제거
    backtest_data['listed_stocks'].fillna(0,inplace=True)
    backtest_data=backtest_data.loc[backtest_data['listed_stocks']!=0]
    
    backtest_data['position']=np.where(backtest_data['factor_ratio']>signal_num,1,0)
    close=pd.pivot_table(backtest_data[['date','code','price']],index='date',values='price',columns=['code'],aggfunc='first')
    market_cap=pd.pivot_table(backtest_data[['date','code','market_cap']],index='date',values='market_cap',columns=['code'],aggfunc='first')
    #position=pd.pivot_table(backtest_data[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first')
    
    universe=set(signal_df.reset_index()['code']).intersection(set(backtest_data['code']))
    position=backtest_data.loc[backtest_data['position']==1]
    
    position=pd.pivot_table(position[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first')
    
    rtn=close.pct_change()
    rtn=rtn.iloc[1:]
    rtn=rtn[universe]
    position=position.shift(1)
    position=position.iloc[1:]
    
    
    
    ###평균적인 주가의 흐름을 보자

    # ticker_date=dict()
    # for code in position.columns:
    #     temp=position[code]
    #     if len(temp.index[(temp.values==1)])>=1:
    #         ticker_date[code]=temp.index[(temp.values==1)]
    
    
    keys=[(index,value) for index,value in zip(position.index, position.columns)]
        
        
    index=[f't{i}' for i in range(minus_end_date,end_date)]
    mean_flow=pd.DataFrame(index=index, columns=[i[1] for i in keys])

    rtn_index=list(rtn.index)

    for key in keys:
        if (pd.to_datetime(rtn.index[-1])-key[0]).days>=end_date:
            mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+end_date],key[1]].values
            if minus_end_date!=0:
                mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].values
        else:
            temp_end_date=end_date
            while (pd.to_datetime(rtn.index[-1])-key[0]).days<temp_end_date:
                temp_end_date-=1
                
            mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+temp_end_date],key[1]].values
            if minus_end_date!=0:
                mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].values
        
        
        
        
        
        
    #mkt=[market_cap.loc[date,code] for date,code in keys]
    mkt=pd.Series([market_cap.loc[date,code] for date,code in keys]).fillna(0)
    mean_flow['market_cap_mean']=np.nan
    for t in mean_flow.index:
        mean_flow.loc[t,'market_cap_mean']=np.dot(mean_flow.loc[t].dropna().values,(mkt/np.sum(mkt)))

    mean_flow['mean']=mean_flow.iloc[:,:-1].mean(axis=1)
    if mean_only==False:
        fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
        fig.show()
    else:
        mean_flow=mean_flow[['mean','market_cap_mean']]
        fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
        fig.show()
    
    return mean_flow