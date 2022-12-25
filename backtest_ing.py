import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from numba import jit
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')



class backtest():
    
    def __init__(self,start,end,data,fee=0,bm=None,factor1=None,factor2=None):
        self.start=start
        self.end=end
        self.bm=bm
        
        self.fee=fee
        self.data=data
        
        
    def cleaning_data(self):
        backtest_data=self.data.copy()
        ###data 차지하는 용량이 너무 큼 삭제
        self.data=0
        ###날짜 맞추기
        backtest_data=backtest_data.loc[(backtest_data['date']>=self.start)&(backtest_data['date']<=self.end)]
        
        ###외부상장 종목 제거
        backtest_data=backtest_data.loc[(backtest_data['market']!='외감') & (backtest_data['market']!='KONEX')]
        ###거래정지 종목 제거
        backtest_data=backtest_data.loc[backtest_data['trading_suspension']!=1]
        ##상장종목수 0인거 제거
        backtest_data['listed_stocks'].fillna(0,inplace=True)
        backtest_data=backtest_data.loc[backtest_data['listed_stocks']!=0]
        
        
        self.close=pd.pivot_table(backtest_data[['date','code','price']],index='date',values='bprice',columns=['code'],aggfunc='first')
        self.market_cap=pd.pivot_table(backtest_data[['date','code','market_cap']],index='date',values='market_cap',columns=['code'],aggfunc='first')
        
        try:
            self.position=pd.pivot_table(backtest_data[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first')
        except Exception as e:
            print(e)
            self.position=0
            
                
            
        if 'factor1' in backtest_data.columns and 'factor2' not in backtest_data.columns:
            self.factor1=pd.pivot_table(backtest_data[['date','code','factor1']],index='date',values='factor1',columns=['code'],aggfunc='first')
            return self.close, self.market_cap, self.position,self.factor1
        
        elif 'factor1' in backtest_data.columns and 'factor2' in backtest_data.columns:
            self.factor1=pd.pivot_table(backtest_data[['date','code','factor1']],index='date',values='factor1',columns=['code'],aggfunc='first')
            self.factor2=pd.pivot_table(backtest_data[['date','code','factor2']],index='date',values='factor2',columns=['code'],aggfunc='first')
            return self.close, self.market_cap, self.position, self.factor1, self.factor2
        
        
        else:       
            return self.close, self.market_cap, self.position
    
    
    def make_rtn(self):
        self.rtn=self.close.pct_change()
        self.rtn-self.rtn.iloc[1:]
        month_end=[]
        for i in range(len(self.close.index)-1):
            temp=self.close.index[i]
            temp2=self.close.index[i+1]
            if str(temp)[5:7]!=str(temp2)[5:7]:
                month_end.append(temp)
        self.month_end=month_end
        price_monthly=self.close.loc[month_end]

        self.monthly_rtn=price_monthly.pct_change()
        self.monthly_rtn=self.monthly_rtn.iloc[1:]
        quarter=[month for month in month_end if str(month)[5:7] in ('3','6','9','12')]
        price_quarterly=self.close.loc[quarter]
        self.quarterly_rtn=price_quarterly.pct_change()
        self.quarterly_rtn=self.quarterly_rtn.iloc[1:]
        years=[year for year in quarter if str(quarter)[5:7]=='12']
        price_yearly=self.close.loc[years]
        self.yearly_rtn=price_yearly.pct_change()
        self.yearly_rtn=self.yearly_rtn.iloc[1:]
        
        return self.rtn, self.monthly_rtn, self.quarterly_rtn, self.yearly_rtn
                
        
    def winsorizing_facotr(self,q,factor):
        
        #temp.clip(upper=temp.T.quantile(q=0.90, interpolation='nearest'),lower=temp.T.quantile(q=0.10, interpolation='nearest'),axis=0)
        factor=factor.clip(upper=factor.quantile(q=q, interpolation='nearest',axis=1),lower=factor.quantile(q=100-q, interpolation='nearest',axis=1),axis=0)
        return factor
        
        
    
    
    #####signal matrix: 가로축 시간, 세로1 code(tickers), 세로2: signal_column
    def make_position(self,signal,signal_column,long_cond,holding_days,short_cond=None,lev_cond=None,lev=1,num=1):
        self.signal=signal[['code',signal_column]]
        self.position=pd.DataFrame(index=self.signal.index,columns=self.signal.columns)
        days=list(self.signal.index)
        for i in range(len(days)-holding_days):
            temp=self.signal.loc[days[i]]
            temp2=temp.set_index('code')
            for code in temp2.index:
                if temp2.loc[code,signal_column]>long_cond*lev_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=lev*num
                elif temp2.loc[code,signal_column]>long_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=num
                elif temp2.loc[code,signal_column]<short_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=-num
                elif temp2.loc[code,signal_column]<short_cond*lev_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=-num*lev
        return self.position.fillna(0)
    
    
    
    def sorting_backtest(self,rebalance_rtn,p,q , width, daily_rtn,vw=True):
        factor_df1=self.factor1.shift(1)
        factor_df1=factor_df1.iloc[1:]
        factor_df2=self.factor_df2.shift(1)
        #market_cap=market_cap.reindex(rebalance_rtn.index)
        market_cap=self.market_cap.loc[self.month_end].shift(1)
        market_cap=market_cap.reindex(factor_df1.index)
        rebalance_rtn=rebalance_rtn.reindex(factor_df1.index)
        factor_df2=factor_df2.reindex(factor_df1.index)
        monthly_cap=market_cap.reindex(factor_df1.index)
        return_tickers=dict()
        
        
        
        #daily_rtn=daily_rtn.loc['2004-05':]
        daily_index=list(daily_rtn.index)
        # monthly_index=list(factor_df1.index)
        
        for i in tqdm(range(0,len(factor_df1))):
            first_sort=factor_df1.iloc[i].dropna().rank(ascending=False,method='first')\
                [(factor_df1.iloc[i].dropna().rank(ascending=False,method='first')>len(factor_df1.iloc[i])*(p-width))&\
                (factor_df1.iloc[i].dropna().rank(ascending=False,method='first')<len(factor_df1.iloc[i])*p)].index.values
            first_sort=list(first_sort)
            second_sort=factor_df2[first_sort]
            
            
            quantile=int(len(second_sort.iloc[i])*q)
            
        
                
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
    
    


    #@jit()
    def run_backtest(self,mode='not_mkt'):
        result=dict()
        self.rtn=self.rtn.reindex(self.position.index)
        ###거래비용이 있는 경우 사전의 포지션에서 거래비용 만큼 제한하고 계산    
        
        for i in tqdm(range(len(self.rtn)-1)):
            
            daily_rtn=self.rtn.iloc[i]
            position=self.position.iloc[i+1]
            position=position.loc[daily_rtn.index]
            temp_position=self.position.iloc[i]
            temp_position=temp_position.loc[position.index]

            
            stocks=[stock for stock in (position.index)]
            position=pd.Series(position,index=stocks)
            temp_position=pd.Series(temp_position,index=stocks)
            
            ###거래비용이 존재하는 경우####
            if self.fee>0:
                for code in temp_position.index:
                    if position.loc[code] != temp_position.loc[code]:
                        daily_rtn.loc[code]=daily_rtn.loc[code]-np.abs(position.loc[code]-temp_position.loc[code])*self.fee
    
            
            if mode=='not_mkt':
                rtn=np.dot(daily_rtn.values,position.values)/((np.sum([weight*position.value_counts()[weight] for weight in position.value_counts().keys()])))
                result[self.rtn.index[i]]=[rtn,stocks]
                
         
            elif mode=='mkt_cap':
                cap=self.market_cap.iloc[i+1]
                cap=cap.loc[daily_rtn.index]
                stocks=[stock for stock in (cap.index)]
                cap=cap*position
                cap=pd.Series(cap,index=stocks)
                rtn=np.dot(daily_rtn.values, np.array([cap.loc[i]/np.sum(cap) for i in cap.index]))
                #result[self.rtn.index[i]]=[rtn,stocks]
                result[self.rtn.index[i]]=[rtn,stocks]    
        
        
        portfolio=pd.DataFrame([i[0] for i in result.values()],index=result.keys(),columns=['return']) 
        if self.bm != None:
            self.bm=np.log(self.bm/ self.bm.shift(1))
            portfolio=pd.concat([portfolio,self.bm],axis=1)
        return portfolio, result
    
    
    def cum_chart(self,portfolio):
        fig=px.line((1+portfolio).cumprod())
        fig.show()
        print(f'누적 수익률 {(1+portfolio).cumprod()}')

        
    def analysis(self,portfolio):
        print("mean","\n",portfolio.mean())
        print("")
        print("std","\n",portfolio.std())
        print("")
        print("skewness","\n",portfolio.skew())
        print("")
        print("kurtosis","\n",portfolio.kurtosis())
        print("")
        print("sharpe ratio","\n", portfolio.mean() / portfolio.std())
        print(f'누적 수익률 {(1+portfolio).prod()}')
        
        
            
    def mdd(self,portfolio):
        cum=(1+portfolio).cumprod()*1000
        arr_v = np.array(cum)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        return peak_upper, peak_lower, f'MDD: {(arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]}'
    
    
    
        
    def turnover(self,result):
        
        turn_over=dict()
        include=dict()
        exclude=dict()
        for i in tqdm(range(1,len(result.keys()))):
            sub=len([ x for x in list(result.values())[i][1] if x not in list(result.values())[i-1][1]] )
            #print(f'교체비율: {sub/len( result[list(result.values())[i-1]][1])}')
            #print('편입',[x for x in result[list(result.values())[i]][1] if x not in result[list(result.values())[i-1]][1]])
            #print('퇴출',[x for x in result[list(result.values())[i-1]][1] if x not in result[list(result.values())[i]][1]])
            turn_over[list(result.keys())[i]]=sub/len(list(result.values())[i-1][1])
            include[list(result.keys())[i]]=[x for x in list(result.values())[i][1] if x not in list(result.values())[i-1][1]]
            exclude[list(result.keys())[i]]=[x for x in list(result.values())[i-1][1] if x not in list(result.values())[i]][1]
            
        turn_over=pd.DataFrame(turn_over.values(),index=turn_over.values(),columns=['turn_over'])
        #include=pd.DataFrame(include.values(),index=include.values(),columns=['include'])
        #exclude=pd.DataFrame(exclude.values(),index=exclude.values(),columns=['exclude'])   
        return turn_over, include, exclude
        
        
    
    def mkt_regression(self,portfolio):
        if self.bm != None:
            X=sm.add_constant(portfolio['return'])
            y=portfolio['bm']
            result=sm.OLS(y,X).fit()
            print(result.summary())
        else: pass



