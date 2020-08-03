import pandas as pd
from dateutil.relativedelta import *
import datetime
import numpy as np
import os
from weasyprint import HTML
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from sklearn.preprocessing import LabelEncoder
import copy
import warnings
warnings.filterwarnings('ignore')

class Report:
    def __init__(self, 
                 metrics,
                 regional_metrics,
                 report_path = '',
                 test_id = ''):
    
        self.all_plots_paths = []
        self.metrics = metrics
        self.regional_metrics  = regional_metrics

        self.test_id = test_id
        self.report_path = report_path
        
    def plot_boxplots(self):
        self.all_plots_paths = []
        for metric, df_metric in self.metrics.groupby('metric'):
            self.all_plots_paths.append(self.plot_overview(df_metric, metric))
            self.all_plots_paths.append(self.plot_bkt_train_oos(df_metric, metric))
        
        return self.all_plots_paths
    
    def plot_overview(self, df, metric_name):
        plt.rcParams['figure.figsize'] = (12, 8)
        df = df[df['period'].isin(['train', 'oos'])]
        
        fig,ax = plt.subplots()
        sns_plt = sns.boxplot(x = 'period', y = 'value', data = df,  palette=("Blues_d"), showmeans = True,
                      meanprops={"marker":"_", "markerfacecolor":"#b50707", "markeredgecolor":"#b50707",  'markeredgewidth' : 2, 'markersize' : 20})
        
        plt.title(metric_name, size = 20)
        plt.xlabel('')
        plt.xticks(size = 16)
        
        plt.ylabel(metric_name, size = 16)
        plt.yticks(size = 12)
        
        path =  self.report_path +  self.test_id + '_{0}_overview.png'.format(metric_name)
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        return path
    
    
    def plot_bkt_train_oos(self, df, metric_name):
        plt.rcParams['figure.figsize'] = (12, 8)
        df = df[df['period'].isin(['train', 'oos'])]
        
        fig,ax = plt.subplots()
        sns_plt = sns.barplot(x = 'safra', y = 'value', hue ='period', data = df,  palette=("Blues_d"))
        
        plt.title(metric_name, size = 20)
        plt.xlabel('Períodos Backtest', size = 16)
        plt.xticks(size = 12, rotation = 45)
        plt.ylabel(metric_name, size = 16)
        plt.yticks(size = 12)
        
        path =  self.report_path +  self.test_id + '_{0}_backtest.png'.format(metric_name)
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        return path
    
    def plot_overview_regional(self, metric_name):
        pass
    
    def plot_bkt_train_oos_regional(self, metric_name):
        pass