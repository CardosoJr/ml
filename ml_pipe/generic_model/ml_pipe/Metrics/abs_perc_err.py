import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime
import numpy as np
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')       
pd.set_option('display.max_columns', 500)

class AbsPercErr:
    def __init__(self, 
                 date_col = 'DATE',
                 model_output = 'PREDICTED',
                 target = 'TARGET', 
                 group_col = 'group',
                 freq = 'm'):
        
        self.date_col = date_col
        self.model_output = model_output
        self.target = target
        self.group_col = group_col
        self.freq = freq
        
        print(self.group_col)
        
        if freq != 'm' and freq != "w": 
            raise ValueError('freq should be m or w')
        
    def __hash__(self):
        return hash(repr(self))
    
    def get_intervals(self, df): 
        min_dt = df[self.date_col].min()
        max_dt = df[self.date_col].max()

        days = (max_dt - min_dt).days

        if self.freq == 'm': 
            total_periods = int(np.ceil(days / 30)) + 3
            sum_periods = [relativedelta(months = 1)] * total_periods
            acc_periods  = np.cumsum(sum_periods)

        else:
            total_periods = int(np.ceil(days / 7)) + 3
            sum_periods = [relativedelta(weeks = 1)] * total_periods
            acc_periods  = np.cumsum(sum_periods)

        bin_periods = [min_dt]
        bin_periods.extend([min_dt + x for x in acc_periods])
        
        df['bin_periods'] = pd.cut(df[self.date_col], bin_periods, right = False, include_lowest = True)
        mapping_periods = dict(zip(list(df['bin_periods'].cat.categories), ['P' +str(x) for x in np.arange(len(bin_periods))]))
        df['bin_periods'] = df['bin_periods'].map(mapping_periods).astype(str)
        
        return df
    
    def get_end_date(self, df, initial_date, num_periods):
        max_dt = df[self.date_col].max() + relativedelta(days = 1) # we are using the last date exclusive, adding one day to facilitate calculations below
        used_periods = num_periods
        while True:
            if self.freq == 'm':
                end_date =  initial_date + relativedelta(months = used_periods)
            elif self.freq == 'w':
                end_date =  initial_date + relativedelta(weeks = used_periods)
            else:
                raise ValueError('freq should be m or w')
                
            used_periods = used_periods - 1    
            
            if (max_dt >= end_date) or (used_periods <= 0):
                break
        
        if (max_dt < end_date) and (used_periods <= 0):
            raise ValueError('dataset does not have one period or more to analyze. Check the dates in your dataset and/or the input analysis period')
                
        return end_date
    
    def calculate(self, df, analysis_periods = {}):
        '''
        description:
            calcualtes elasticity error metric
        
        input:
            df : dataframe with predictions and targets 
            analysis_periods: 'period_name' : (first_date / inclusive, end_date / exclusive) 
        output 
            dataframe with metric values
        '''
        
        
        
        all_err = pd.DataFrame([])
        for name, period in analysis_periods.items():
            end_date = self.get_end_date(df, period[0], period[1])

            df_aux = df[(df[self.date_col] >= period[0]) & (df[self.date_col] < end_date)]
            
            err = self._calculate(df_aux)
            err['period'] = [name] * len(err)
            all_err = all_err.append(err, ignore_index = True)
    
        return all_err
    
    def _calculate(self, df):
        '''
        description:
            calcualtes absolute percentage error metric
        input:
            df : dataframe with predictions and targets 
            params: parameters for metric calculation 
        output 
            dataframe with metric values
        
        '''
        df = self.get_intervals(df)
        predicted = df.groupby(['bin_periods', self.group_col])[self.model_output].agg(['mean','count']).rename(columns={'mean':self.model_output}).reset_index()
        
        real = df.groupby(['bin_periods', self.group_col])[self.target].mean().reset_index()
        final =  real.merge(predicted, on = ['bin_periods', self.group_col])
        final = final.dropna()
        err = pd.DataFrame([])
        for period, df_period in final.groupby('bin_periods'):
            for region, df_region in df_period.groupby(self.group_col):
                if 'ZZ' in region:
                    continue
                metric = pd.DataFrame([])
                metric['count'] = df_region['count']
                metric['value'] = [np.mean(np.abs(df_region[self.target] - df_region[self.model_output]) * 100.0 / df_region[self.target])]
                metric[self.group_col] = [region] * len(metric)
                metric['bin_periods'] = [period] * len(metric)
                err = err.append(metric, ignore_index = True)
        err['metric'] = ['perc_err'] * len(err)
        return err
   