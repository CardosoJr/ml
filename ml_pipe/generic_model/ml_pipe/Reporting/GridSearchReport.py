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
import pickle
import warnings
warnings.filterwarnings('ignore')
import itertools

class Report:
    def __init__(self, 
                 opt_file,
                 report_path = '',
                 test_id = '',
                 grouping = [True, False]):
    
        self.grouping = grouping
        self.opt_file = opt_file
        with open(opt_file, "rb") as f:
            self.tpe_trials = pickle.load(f)
        self.test_id = test_id
        self.report_path = report_path

    def plot_optimization_overview(self):
        plt.rcParams['figure.figsize'] = [12, 9]
        losses = [x['result']['loss'] for x in self.tpe_trials.trials]
        min_loss = list(itertools.accumulate(losses, min))

        ax = plt.subplot(111)
        plt.plot(np.arange(len(losses)), losses, '--', color = 'grey')
        plt.plot(np.arange(len(losses)), min_loss, 'k.-')
        plt.xticks(size = 12)
        plt.yticks(size = 12)
        plt.xlabel('iterations', size = 14)
        plt.ylabel('Metrica', size = 16)
        path =  self.report_path +  self.test_id + '_gs_overview.png'
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()
        
    def plot_k_best(self, k):
        loss_par = list(set([(x['result']['loss'], str(x['misc']['vals'])) for x in self.tpe_trials.trials]))
        loss_par = sorted(loss_par, key = lambda tup: tup[0])
        values = [x[0] for x in loss_par][:k]
        pars = [str(x[1]).replace(',', '\n').replace('[', '').replace(']', '').replace('.0', '') for x in loss_par][:k]
        plt.rcParams['figure.figsize'] = [20, 15]
        sns.barplot(pars, values, palette=("Blues_d"))
        plt.ylim(np.min(values)*0.999, np.max(values))
        plt.xticks(size = 12)
        plt.ylabel('Métrica', size = 16)
        path =  self.report_path +  self.test_id + '_gs_k_best.png'
        plt.savefig(path, bbox_inches ='tight', pad_inches = 0 )
        plt.show()    
        

        
        
        