import pandas as pd
import numpy as np

####market_cap은 한달 주기의 시가총액을 넣을 것
def sorting_backtest(rebalance_rtn,factor_df1,factor_df2,p,q , width, daily_rtn, market_cap,vw=True):
    factor_df1=factor_df1.shift(1)
    factor_df1=factor_df1.iloc[1:]
    factor_df2=factor_df2.shift(1)
    #market_cap=market_cap.reindex(rebalance_rtn.index)
    market_cap=market_cap.shift(1)
    monthly_cap=market_cap.reindex(factor_df1.index)
    rebalance_rtn=rebalance_rtn.reindex(factor_df1.index)
    factor_df2=factor_df2.reindex(factor_df1.index)
    #monthly_cap=market_cap.reindex(factor_df1.index)
    return_tickers=dict()
    
    
    
    #daily_rtn=daily_rtn.loc['2004-05':]
    daily_index=list(daily_rtn.index)
    # monthly_index=list(factor_df1.index)
    
    for i in range(0,len(factor_df1)):
        first_sort=factor_df1.iloc[i].dropna().rank(ascending=False,method='first')\
            [(factor_df1.iloc[i].dropna().rank(ascending=False,method='first')>len(factor_df1.iloc[i])*(p-width))&\
            (factor_df1.iloc[i].dropna().rank(ascending=False,method='first')<len(factor_df1.iloc[i])*p)].index.values
        first_sort=list(first_sort)
        second_sort=factor_df2[first_sort]
        
        
        quantile=int(len(second_sort.iloc[i])*q)
        last_quantile=int(len(second_sort.iloc[i]))-quantile
    
            
        stocks_selected=second_sort.iloc[i].dropna().rank(ascending=False,method='first')\
            [(second_sort.iloc[i].dropna().rank(ascending=False,method='first')>int(len(second_sort.iloc[i])*(q-width)))&
            (second_sort.iloc[i].dropna().rank(ascending=False,method='first')<quantile)].index.values
            
        start=daily_index.index(factor_df1.index[i-1])
        if i!=1:
            temp_daily=daily_rtn.loc[daily_index[start+1]:factor_df1.index[i]]
        else:
            temp_daily=daily_rtn.loc[:factor_df1.index[i-1]]
        
        if vw==True:
            
            cap=monthly_cap.iloc[i]
            cap=cap.reindex(stocks_selected)
            cap=cap.dropna()
           
            
            for day in temp_daily.index:
                return_tickers[day] = \
                [[np.dot(daily_rtn.loc[day].reindex(cap.index).values, cap/np.sum(cap))], cap.index]
        
        
        
        else:
            for day in temp_daily.index:
                
                return_tickers[day] = \
            [[np.dot(daily_rtn.loc[day].reindex(stocks_selected).values, np.array([1/len(stocks_selected) for i in range(len(stocks_selected))]))], stocks_selected]
      
    
    
    
    return return_tickers
        
        
        
