import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime
import numpy as np
import os
import shutil
from sklearn import metrics
import copy
from .auc import Auc


class ElasticityErr(Auc):
    def __init__(self, 
                 num_bins = 100, 
                 date_col = 'DATE',
                 model_output = 'PREDICTED',
                 target = 'TARGET', 
                 group_col = 'group',
                 freq = 'm'):
        self.num_bins = num_bins
        super().__init__(date_col, model_output, target, group_col, freq)
    
    def calculate_metric(self, y_test, y_pred):
        raise Exception('method not implemented')
    
    def define_metric_name(self):
        return 'elasticity_err'
    
    def calculate_per_group(self, df):
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
        err['metric'] = [self.define_metric_name()] * len(err)
        return err[[self.group_col, 'value', 'metric', 'bin_periods','count']]
        