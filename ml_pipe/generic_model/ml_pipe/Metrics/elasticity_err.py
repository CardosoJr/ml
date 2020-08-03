import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime
import numpy as np
import os
pd.set_option('display.max_columns', 500)
import shutil
from sklearn import metrics
import copy
import warnings
warnings.filterwarnings('ignore')   

class ElasticityErr:
    def __init__(self, 
                 num_bins = 100, 
                 date_col = 'DATE',
                 model_output = 'PREDICTED',
                 target = 'TARGET', 
                 group_col = 'group',
                 freq = 'm'):
        self.num_bins = num_bins
        self.date_col = date_col
        self.model_output = model_output
        self.target = target
        self.group_col = group_col
        self.freq = freq
        
        if freq != 'm' and freq != "w": 
            raise ValueError('freq should be m or w')
    
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
        err = pd.DataFrame([])
        
        df = self.get_intervals(df)
        df['RAZAO'] = df['VAL_PR_ANUAL_LIQUIDO_RENOV'] / df['VAL_PR_ANUAL_LIQUIDO']
        bin_var = 'RAZAO'
        clip_min = 0.5
        clip_max = 2.0
        df[bin_var] = df[bin_var].clip(clip_min, clip_max)
        bins = np.linspace(df[bin_var].min(), df[bin_var].max(), self.num_bins)
        try:
            df['binned'] = pd.cut(df[bin_var], bins)
        except:
            err = pd.DataFrame([])
            err[self.group_col] = [x for x in list(df[self.group_col].unique()) if 'ZZ' not in x]
            err['value'] = [np.nan] * len(err)
            err['metric'] = ['elasticity_err'] * len(err)
            return err
        
        for region, df_region in df.groupby(self.group_col):
            if 'ZZ' in region:
                continue 
            region_count = df_region.groupby(['bin_periods'])[self.target].agg('count')
            group = df_region.groupby(['binned', 'bin_periods'])
            elasticity = pd.DataFrame([])
            elasticity['COUNT'] = group[self.target].count()
            elasticity['MEAN_DEMAND'] = group[self.target].mean()
            elasticity['PREDICTED_MEAN_DEMAND'] = group[self.model_output].mean()
            elasticity = elasticity.reset_index()
            ## ERRORS
            elasticity['MEAN_PERC_ERR'] = 2 * np.abs(elasticity['PREDICTED_MEAN_DEMAND'] - elasticity['MEAN_DEMAND']) / np.maximum(np.abs(elasticity['PREDICTED_MEAN_DEMAND']), np.abs(elasticity['MEAN_DEMAND']))
            elasticity =  elasticity.merge(region_count, on = ['bin_periods'])
            
            el_err = []
            anomes_indexes = []
            count_values = []
            for anomes, df_anomes in elasticity.groupby('bin_periods'):
                ## ERRORS
                df_anomes['value'] = (df_anomes['MEAN_PERC_ERR'] * df_anomes['COUNT']) / df_anomes['COUNT'].sum() 
                el_err.append(df_anomes['value'].sum())
                anomes_indexes.append(anomes)
                count_values.append(region_count.loc[anomes])
                            
            all_results_grouped = pd.DataFrame(list(zip(anomes_indexes, el_err, count_values)), 
                                               columns = ['bin_periods', 'value','count'])
            all_results_grouped[self.group_col] = [region] * len(all_results_grouped) 
            
            err = err.append(all_results_grouped, ignore_index = True)   
        err['metric'] = ['elasticity_err'] * len(err)
        return err[[self.group_col, 'value', 'metric', 'bin_periods','count']]
        