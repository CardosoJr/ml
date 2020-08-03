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
                 training_length = 1,
                 lead_time = 4,
                 lead_time_mode = 'w', # in week | 'm' for monthly lead time
                 report_path = '',
                 base_dir ='', 
                 test_id = '')
    
        self.all_plots_paths = []
        self.df_results = {}
        
        self.training_length = training_length
        self.lead_time = lead_time
        self.lead_time_mode = lead_time_mode

        self.test_id = test_id
        self.base_dir = base_dir
        self.report_path = report_path

    

    

    def plot_auc_boxplot(self):
        plt.rcParams['figure.figsize'] = (12, 8)
        fig,ax = plt.subplots()
        sns_plt = sns.boxplot(x = 'week', y = 'auc_roc', data = self.general_auc_roc,  palette=("Blues_d"), showmeans = True,
                      meanprops={"marker":"_", "markerfacecolor":"#b50707", "markeredgecolor":"#b50707",  'markeredgewidth' : 2, 'markersize' : 20})
        path =  self.report_path +  self.test_id + '_auc_boxplot.png'
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        return path
    
    def plot_err_boxplots(self):
        plt.rcParams['figure.figsize'] = (12, 8)
        fig,ax = plt.subplots()
        sns_plt = sns.boxplot(x = 'week', y = 'week_perc_err', data = self.general_perc_errs,  palette=("Blues_d"), showmeans = True,
                      meanprops={"marker":"_", "markerfacecolor":"#b50707", "markeredgecolor":"#b50707",  'markeredgewidth' : 2, 'markersize' : 20})
        path =  self.report_path +  self.test_id + '_boxplot.png'
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        return path
    
    def plot_err_reg_boxplots(self):
        plt.rcParams['figure.figsize'] = (20, 8)
        fig,ax = plt.subplots()
        sns_plt = sns.boxplot(x = 'week', y = 'week_perc_err', hue = 'REGIONAL', data = self.perc_errs)
        path =  self.report_path +  self.test_id + '_reg_boxplot.png'
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        return path