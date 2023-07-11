import numpy as np
import pandas as pd
import scipy as sp
from numba import njit
import datetime
import plotly.express as px





class simple_backtset():
    def __init__(self, data,start,end,fee=0.005):
        
        self.start=pd.to_datetime(start)
        self.end=pd.to_datetime(end)
        self.temp_data=data.loc[(data['date']>=self.start)&(data['date']<=self.end)]
        self.fee=fee

    def mean_stock_movement(self,signal_num, end_date,factor_name,minus_end_date=0,mean_only=False,lagging=1):
        try:
            data=data.rename(columns={"adj_close":'price'})
        except Exception as e:
            pass
        # universe=set(new_sue.reset_index()['level_1'])
    
        
        ###외부상장 종목 제거
        backtest_data=self.temp_data.loc[(self.temp_data['market']!='외감') & (self.temp_data['market']!='KONEX')]
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
    
        rtn_index=list(rtn.index)
        keys=[(index,code)  for index in key_df.index[1:] for code in universe if (key_df.loc[index,code]>=1) and (rtn_index.index(index)>=np.abs(minus_end_date)) and  (  (rtn_index.index(rtn_index[-1])-rtn_index.index(index)) > end_date)]
    
    
        index=[f't{i}' for i in range(minus_end_date,end_date)]
        mean_flow=pd.DataFrame(index=index, columns=set([i[1] for i in keys]))
    
        #rtn_index=list(rtn.index)
    
        #keys=[(key[0],key[1]) for key in keys if (rtn_index.index(key[0])>=np.abs(minus_end_date)) and  (  (rtn_index.index(rtn_index[-1])-rtn_index.index(key[0])) > end_date)]
    
        for key in tqdm(keys):
            
    
            try:
                mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+end_date],key[1]].values
            except Exception as e:
                mean_flow.loc['t0':f't{end_date-1}',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0]):rtn_index.index(key[0])+end_date],key[1]].fillna(method='ffill').values
            
            
    
            if rtn_index.index(key[0])+minus_end_date>0:
                try:
                    mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].values
                except Exception as e:
                    mean_flow.loc[f't{minus_end_date}':'t-1',key[1]]=rtn.loc[rtn_index[rtn_index.index(key[0])+minus_end_date:rtn_index.index(key[0])],key[1]].fillna(method='ffill').value
                    
    
    
    
    
        #mkt=[market_cap.loc[date,code] for date,code in keys]
        mean_flow.dropna(axis=1,how='all',inplace=True)
        mean_flow['mean']=mean_flow.iloc[:,:-1].mean(axis=1,skipna=True)
        if mean_only==False:
            self.mean_flow=mean_flow
            fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
            fig.show()
        else:
            mean_flow=mean_flow['mean']
            self.mean_flow=mean_flow
            fig=px.line((1+mean_flow.fillna(0)).cumprod()-(1+mean_flow.loc[:'t0']).cumprod().loc['t0'])
            fig.show()
            return mean_flow
        
    

    def restrict_cap(self,top=500):
        self.temp_data=self.temp_data.set_index(['date'])
        self.temp_data['mkt_rank']=np.nan
        for ind in sorted(list(set(self.temp_data.index))):
            self.temp_data.loc[ind,'mkt_rank']=self.temp_data.loc[ind,'market_cap'].rank(ascending=False,method='first')
    
        self.temp_data=self.temp_data.reset_index()
        self.temp_data=self.temp_data.loc[self.temp_data['mkt_rank']>=top]
        return self.temp_data

    def get_position(self, **cond):
        """
        
        ----------
        Params :
        upper_Condition = condition[0] 상한 이거보다 작아야 진입
        longer_Condition = condition[1] 하한 이거보다 커야 진입 
        Is_Short_Strg? (T or F) = condition[2] 공매도 전략 여부
        Enter_Weight  = condition[3] 초기 투자 비중
        
        Holding_Period(days) = condition[4] 보유기간
        Enter_Lag time(days) = condition[5]  신호 후 며칠 후 진입인지
        
        ----------
        **cond : LIST
        """
        ###외부 시장 및 코넥스 제거
        self.temp_data=self.temp_data.loc[(self.temp_data['market']!='외감') & (self.temp_data['market']!='KONEX')]
        ###거래정지 종목 제거
        self.temp_data=self.temp_data.loc[self.temp_data['trading_suspension']!=1]
        ##상장종목수 0인거 제거
        self.temp_data['listed_stocks'].fillna(0,inplace=True)
        self.temp_data=self.temp_data.loc[self.temp_data['listed_stocks']!=0]
        self.temp_data['rtn']=np.log(self.temp_data.groupby('code').adj_close.shift(0)/self.temp_data.groupby('code').adj_close.shift(1))   
        self.temp_data['rtn_1']=self.temp_data.groupby('code').rtn.shift(1)
        ###혹시나 모를 오류 제거, 상한가 하한가 30% 이상 차이나는 종목들 제거
        self.temp_data=self.temp_data.loc[(self.temp_data['rtn']<=0.3) & (self.temp_data['rtn']>=-0.3)]
        
        
        ####월말, 분기말 연말 날짜 구하기
      
                
        
        t=1
        for factor, cond in cond.items():
            factor=f"{factor}"
            upper_cond=cond[0]
            lower_cond=cond[1]
            is_short=cond[2]
            invest_ratio=cond[3]
            holding_days=cond[4]
            lag=cond[5]
            position = -1 if (is_short is True) | (is_short == -1) | (is_short == "S") else 1
            
            
            
            
            # ####월말, 분기말 연말 날짜 구하기
            monthend=list()
            if type(holding_days)==str:
                days=sorted(list(set(self.temp_data['date'])))
        
                for day in range(len(days)-1):
                    if str(days[day])[5:7]!=str(days[day+1])[5:7]:
                        monthend.append(days[day])
                
                if holding_days in ['quarter', 'Q','q']:
                    quarter_end=[]
                    for day in monthend:
                        if str(day)[5:7] in ['03','06','09','12']:
                            quarter_end.append(day)
                            
                    monthend=quarter_end
                else:
                    year_end=[]
                    for day in monthend:
                        if str(day)[5:7] in ['12']:
                            year_end.append(day)
                            
                    monthend=year_end
                    
               
              
                
            ################################################
            
            
           
            ####각각의 진입 시기만 남기기 
            if (lower_cond is not None) & (upper_cond is not None):
                     
                sign_idx = np.where((self.temp_data[factor] >= lower_cond) & (self.temp_data[factor]< upper_cond),invest_ratio*position,0)
                        
            elif lower_cond is not None:
                
            
                if (type(lower_cond) == int) | (type(lower_cond) == float):
                    sign_idx = np.where((self.temp_data[factor] >= lower_cond),invest_ratio*position,0)
                else:
                    print("Lower Condition Error")
                    sign_idx = None
                
            elif upper_cond is not None:
                    
                if (type(upper_cond) == int) | (type(upper_cond) == float):
                    sign_idx = np.where((self.temp_data[factor] < upper_cond),invest_ratio*position,0)
                else:
                    print("Higher Condition Error")
                    sign_idx = None
                
            else:
                print("No Condition")
                sign_idx = None

            self.temp_data[f'factor_{t}']=sign_idx
            
              ###lagging 후진입
            
            self.temp_data[f'factor_{t}']=self.temp_data.groupby('code')[f'factor_{t}'].shift(lag)
            ##보유기간이 일수로 정해져 있는경우
            if type(holding_days)==int or  type(holding_days)==float:
                self.temp_data[f'factor_{t}_clear']=self.temp_data.groupby('code')[f'factor_{t}'].shift(holding_days)
                self.temp_data[f'factor_{t}_clear']=self.temp_data[f'factor_{t}_clear']*-1
                self.temp_data[f'factor_{t}'].fillna(0,inplace=True)
                self.temp_data[f'factor_{t}_clear'].fillna(0,inplace=True)
                self.temp_data[f'pos_{t}']=self.temp_data[f'factor_{t}']+self.temp_data[f'factor_{t}_clear']
            
            ### 보유기간이 진입 후 한달, 3개월, 6개월, 1년인 경우
            else:
                self.temp_data_month=self.temp_data.set_index('date')
                self.temp_data_month=self.temp_data_month.loc[monthend]
                self.temp_data_month['factor_exit']=self.temp_data_month[f'factor_{t}'].groupby('code').shift(1)
                self.temp_data[f'factor_{t}_clear']=self.temp_data_month['factor_exit']*-1
                self.temp_data[f'factor_{t}'].fillna(0,inplace=True)
                self.temp_data[f'factor_{t}_clear'].fillna(0,inplace=True)
                self.temp_data[f'pos_{t}']=self.temp_data[f'factor_{t}']+self.temp_data[f'factor_{t}_clear']
                
                    
                    
                    
                    
            t+=1
        ###포지션 만들기####
        
        factor_list=[col for col in self.temp_data.columns if col.startswith('pos')]
        ###멅티전략 포지션 array 합치기, 상계처리
        self.temp_data['position']=np.nansum(self.temp_data[factor_list],axis=1)
            
            
            
            
            
         
  
       
            
            
            
        daily_rtn=pd.pivot_table(self.temp_data[['date','code','rtn']],index='date',values='rtn',columns=['code'],aggfunc='first',dropna=False)
        position=pd.pivot_table(self.temp_data[['date','code','position']],index='date',values='position',columns=['code'],aggfunc='first',dropna=False)
        #position=position.shift(1)
        position.fillna(0,inplace=True)
        position=position.cumsum()
        daily_rtn=daily_rtn.reindex(position.index)
        
        self.date_index=list(position.index)
        #daily_rtn=daily_rtn.iloc[1:]
        self.fee_matrix=(position.diff())*self.fee
        self.position=np.nan_to_num(position.values)
        self.daily_rtn=np.nan_to_num(daily_rtn.values)
        self.fee_matrix=self.fee_matrix.values
        
        
        
        
        return self.position, self.daily_rtn
      
        
        
        
        
        
    @njit()
    def simulation(self,fee=0.005,nav=1):
  
        
        nav_signal=self.position.copy()
        position_signal=self.position.copy()
        #fee_matrix=np.diff(position_signal,axis=1)*fee

        #nav_signal=nav_signal-(fee_matrix*np.sign(nav_signal))
        nav_result=[np.float(nav)]
        for i in range(len(position_signal)-1):    
            nav_signal[i+1]+=np.abs(nav_signal[i])*self.daily_rtn[i]-np.nansum(np.abs(self.fee_matrix[i]))
            nav+=np.nansum(nav_signal[i]*self.daily_rtn[i])-np.nansum(np.abs(self.fee_matrix[i]))
            #print(np.nansum(nav_signal.iloc[i].fillna(0)*daily_rtn.iloc[i]))
            nav_result.append(nav)
        self.simple_result=nav_result
            

        return self.simple_result
    
    
    
    
    def position_count(self):
        num_list=[]
        long_num_list=[]
        short_num_list=[]
        pos=pd.DataFrame(self.position,index=self.date_index)
        for date in self.date_index:
            temp=pos.loc[date]
            num_list.append(len(temp.loc[temp!=0].dropna()))
            long_num_list.append(len(temp.loc[temp>0].dropna()))
            short_num_list.append(len(temp.loc[temp<0].dropna()))
            
        self.count_df=pd.DataFrame(index=self.date_index)
        self.count_df['gross']=num_list
        self.count_df['long']=long_num_list
        self.count_df['short']=short_num_list
        fig=px.line(self.count_df)
        fig.show()
        return self.count_df  
    
    
    
    
    
    
    def analysis(self):
        arr_v=np.array(self.simple_result)
        peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
        peak_upper = np.argmax(arr_v[:peak_lower])
        MDD=(arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]
        print(f"MDD: {MDD}")
        rtn=(self.simple_result[-1])-1
        ###연율화 수익률
        rtn_annualy=(1+rtn)**(252/len(self.simple_result))-1
      
        ### 테스트 기간동안의 일일 총 평균 투자비중(gross 투자 비중)
        mean_invest_ratio=np.mean(np.sum(np.abs(self.position),axis=1))
        ### 테스트 기간동안의 일일 최대 투자비중(gross 투자 비중)
        max_invest_ratio=np.max(np.sum(np.abs(self.position),axis=1))
        print(f"최종 누적수익률: {rtn*100}%")
        print(f"수익률/평균투자비중: {rtn/mean_invest_ratio}")
        print(f"수익률/최대투자비중: {rtn/max_invest_ratio}")
        print(f"연환산수익률/평균투자비중: {rtn_annualy/mean_invest_ratio}")
        self.result=pd.DataFrame(data=self.simple_result,index=self.date_index)
        fig=px.line(self.result)
        fig.show()
        
        



