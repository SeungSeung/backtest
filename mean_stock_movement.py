import pandas as pd
from sqlalchemy import create_engine
import pymysql
import numpy as np
import plotly.express as px
from tqdm import tqdm


def mean_stock_movement(signal_num,data, end_date,factor_name,minus_end_date=0,mean_only=False,lagging=1):
    try:
        data=data.rename(columns={"adj_close":'price'})
    except Exception as e:
        pass
    # universe=set(new_sue.reset_index()['level_1'])

    
    ###외부상장 종목 제거
    backtest_data=data.loc[(data['market']!='외감') & (data['market']!='KONEX')]
    ###거래정지 종목 제거
    backtest_data=backtest_data.loc[backtest_data['trading_suspension']!=1]
    ##상장종목 0인거 제거
    backtest_data['listed_stocks'].fillna(0,inplace=True)
    backtest_data=backtest_data.loc[backtest_data['listed_stocks']!=0]

    backtest_data['position']=np.where(backtest_data[factor_name]>signal_num,1,0)

    close=pd.pivot_table(backtest_data[['date','code','price']],index='date',values='price',columns=['code'],aggfunc='first',dropna=False)
    #market_cap=pd.pivot_table(backtest_data[['date','code','market_cap']],index='date',values='market_cap',columns=['code'],aggfunc='first')
    #position=pd.pivot_table(backtest_data[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first')

    #universe=set(universe).intersection(set(backtest_data['code']))
    position=backtest_data.loc[backtest_data['position']==1]
    universe=list(set(position['code']))
    close = close.loc[:,~close.columns.duplicated()].copy()
    key_df=pd.pivot_table(backtest_data[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first',dropna=True)
    #position=pd.pivot_table(position[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first')

    rtn=close.pct_change()
    rtn=rtn.iloc[1:]
    rtn=rtn[universe]
    key_df=key_df.shift(lagging)
    key_df=key_df.iloc[lagging:]



    ###평균적인 주가의 흐름을 보자


    keys=[(index,code)  for index in key_df.index[1:] for code in universe if key_df.loc[index,code]>=1]


    index=[f't{i}' for i in range(minus_end_date,end_date)]
    mean_flow=pd.DataFrame(index=index, columns=set([i[1] for i in keys]))

    rtn_index=list(rtn.index)

    for key in tqdm(keys):
        b_day=np.busday_count(enddates=str(rtn.index[-1])[:10], begindates=str(key[0])[:10])
        if b_day>end_date:
            try:
                mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+end_date],key[1]].values
            except Exception as e:
                mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+end_date],key[1]].fillna(method='ffill').values
            
            
            if minus_end_date!=0:
                if rtn_index.index(key[0])+minus_end_date>0:
                    try:
                        mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].values
                    except Exception as e:
                        mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].fillna(method='ffill').values
                else:
                    temp_minus_end_date=minus_end_date
                    while rtn_index.index(key[0])<temp_minus_end_date*-1:
                        temp_minus_end_date+=1
                    
                    mean_flow.loc[f't{temp_minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+temp_minus_end_date:rtn_index.index(key[0])],key[1]].values
        else:
            temp_end_date=end_date
            if b_day<temp_end_date:
                temp_end_date=b_day

            try:
        ####기간이 테스트 기간 끝까지 해도 받은 기간보다 짧은 종목은 패스
        
                mean_flow.loc['t0':f't{temp_end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+temp_end_date],key[1]].values
    
            except Exception as e:
                pass
            try:    
                mean_flow.loc['t0':f't{temp_end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+temp_end_date],key[1]].fillna(method='ffill').values
            except Exception as e:
                mean_flow.loc['t0':f't{temp_end_date-1}',key[1]]=np.nan
            
            
            
            
            
            if minus_end_date!=0:
                if rtn_index.index(key[0])+minus_end_date>0:
                    try:
                        mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].values
                    except Exception as e:
                        mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=np.nan
                else:
                    temp_minus_end_date=minus_end_date
                    while rtn_index.index(key[0])<minus_end_date*-1:
                        temp_minus_end_date+=1
                    mean_flow.loc[f't{temp_minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+temp_minus_end_date:rtn_index.index(key[0])],key[1]].values
                











    #mkt=[market_cap.loc[date,code] for date,code in keys]
    mean_flow.dropna(axis=1,how='all',inplace=True)
    mean_flow['mean']=mean_flow.iloc[:,:-1].mean(axis=1,skipna=True)
    if mean_only==False:
        fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
        fig.show()
    else:
        mean_flow=mean_flow['mean']
        fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
        fig.show()
        return mean_flow
