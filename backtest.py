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
    
    def __init__(self,position,rtn,start,end,suspending,fee=0,bm=None,mkt=None):
        self.start=start
        self.end=end
        self.bm=bm
        self.mkt=mkt
        self.rtn=rtn.loc[self.start:self.end]
        self.position=position.loc[self.start:self.end]
        self.suspending=suspending.loc[self.start:self.end]
        self.fee=fee
    
    #####signal matrix: 가로축 시간, 세로1 code(tickers), 세로2: signal_column
    def make_position(self,signal,signal_column,long_cond,holding_days,short_cond=None,lev_cond=None,lev=1,num=1):
        self.signal=signal[['code',signal_column]]
        self.position=pd.DataFrame(index=self.signal.index,columns=self.signal.columns)
        days=list(self.signal.index)
        for i in range(len(days)-holding_days):
            temp=self.signal.loc[days[i]]
            temp2=temp.set_index('code')
            for code in temp2.index:
                threshold=temp2.loc[code,signal_column]
                if threshold>long_cond*lev_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=lev*num
                elif threshold>long_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=num
                elif threshold<short_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=-num
                elif threshold<short_cond*lev_cond:
                    self.position.loc[days[i]:days[i+holding_days],code]=-num*lev

        return self.position.fillna(0)


    #@jit()
    def run_backtest(self,mode='not_mkt'):
        result=dict()
        self.rtn=self.rtn.reindex(self.position.index)
         
        
        for i in tqdm(range(len(self.rtn)-1)):
            stop=self.suspending.iloc[i+1]
            daily_rtn=self.rtn.iloc[i]
            "거래정지항목 제외"
            stop=stop.loc[stop.values != 1].index
            
            '''거래가능하고 유니버스에 포함된 종목'''
            
            universe=list(set(stop).intersection(set(daily_rtn.index)))
            daily_rtn=daily_rtn.loc[universe]
            
            ####거래정지 지우면 거의다 na값 리턴을 없앨 수 있지만 혹시나 모르기 때문에
            
            daily_rtn=daily_rtn.dropna()
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
                cap=self.mkt.iloc[i+1]
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



