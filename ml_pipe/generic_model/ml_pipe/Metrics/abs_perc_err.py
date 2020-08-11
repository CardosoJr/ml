import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime
import numpy as np
from sklearn import metrics

from .auc import Auc

class AbsPercErr(Auc):
    def __init__(self, 
                 date_col = 'DATE',
                 model_output = 'PREDICTED',
                 target = 'TARGET', 
                 group_col = 'group',
                 freq = 'm'):
        super().__init__(date_col, model_output, target, group_col, freq)
        
    def __calculate_metric(self, y_test, y_pred):
        raise Exception('method not implemented')
    
    def __define_metric_name(self):
        return 'perc_err'
    
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
        err['metric'] = [self.metric_name] * len(err)
        return err
   