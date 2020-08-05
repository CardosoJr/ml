import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc
pd.set_option('display.max_columns', 500)
from datetime import datetime
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from google.cloud import storage
from datetime import datetime
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime
from datetime import timedelta  
from dateutil.relativedelta import relativedelta

from .CreateDS import CreateDataSet


class CreateMultipleDatasets:
    def __init__(self, run = True):
        self.run = run

    def set_run(self, run = True):
        self.run = run
        
    def generate_datesets(self, 
                          features, 
                          series_columns,
                          initial_training_month, 
                          last_predicting_month,
                          label , 
                          table_name,
                          lead_time = 4, 
                          lead_time_mode = 'w',
                          training_length = 3, 
                          test_size = 3, 
                          base_dir = '',
                          ds_name = 'test',
                          analysis_variables = []):
        if self.run:
            self.run = False
            feature_label =  features
            feature_label.append(label)
            
            all_paths = {}
            train_begin = initial_training_month
            train_end = train_begin + relativedelta(months = training_length)
            
            if lead_time_mode == 'w':
                prediction_month = train_end + relativedelta(weeks = lead_time)
            else:
                prediction_month = train_end + relativedelta(months = lead_time)
            
            test_begin = train_begin
            test_end = train_end + relativedelta(months = test_size)    
            
            while prediction_month <= last_predicting_month:
                
                print('\t\tDS Train: {0} - {1} | Test : {2} - {3}'.format(train_begin.strftime("%Y-%m-%d"), 
                                                                          train_end.strftime("%Y-%m-%d"),
                                                                          test_begin.strftime("%Y-%m-%d"),
                                                                          test_end.strftime("%Y-%m-%d")))
                
                path = base_dir + "{0}/{1}/".format(ds_name, train_begin.strftime("%Y%W"))
                vehicle_categ = "10, 11, 14, 15, 20, 21"
                
                dsgen = CreateDataSet()
                params = {
                            'period_begin' : train_begin.strftime("%Y-%m-%d"),
                            'period_end' : train_end.strftime("%Y-%m-%d"),
                            'period_test_end' :  test_end.strftime("%Y-%m-%d"),
                            'train_size' : 0.8,
                            'target' : label,
                            'cols' : feature_label,

                            'query' : """
                             select {0}
                            from {1} """.format(','.join(feature_label), table_name),

                            'where' : """ WHERE  CATEGORIA in ({2})   
                            """.format(train_begin.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"), vehicle_categ),

                            'table_name' : table_name,
                            'dir' : path,
                            'analysis_variables' : analysis_variables
                        }
                
                ds_paths = dsgen.create_dataset_bq(**params)
                all_paths[train_end.strftime("%Y%W")] = ds_paths
                
                if lead_time_mode == 'w':
                    train_begin = train_begin + relativedelta(weeks = lead_time)
                else:
                    train_begin = train_begin + relativedelta(months = lead_time)
                    
                train_end = train_begin + relativedelta(months = training_length)
                
                if lead_time_mode == 'w':
                    prediction_month = train_end + relativedelta(weeks = lead_time)
                else:
                    prediction_month = train_end + relativedelta(months = lead_time)
                    
                test_begin = train_begin
                test_end = train_end + relativedelta(months = test_size)

            return all_paths
        else:
            return self.get_ds_paths(initial_training_month, last_predicting_month, lead_time, lead_time_mode,training_length,test_size, base_dir, ds_name)

    def __hash__(self):
        return hash(repr(self))
    
    def get_ds_paths(self,
                     initial_training_month, 
                     last_predicting_month, 
                     lead_time = 4, 
                     lead_time_mode = 'w',
                     training_length = 3,
                     test_size = 3,
                     base_dir = '', 
                     ds_name = 'test'):
        
        all_paths = {}
        train_begin = initial_training_month
        train_end = train_begin + relativedelta(months = training_length)
        
        if lead_time_mode == 'w':
            prediction_month = train_end + relativedelta(weeks = lead_time)
        else:
            prediction_month = train_end + relativedelta(months = lead_time)
            
        test_begin = train_begin
        test_end = train_end + relativedelta(months = test_size)
        
        while prediction_month <= last_predicting_month:
            path = base_dir + "{0}/{1}/".format(ds_name, train_end.strftime("%Y%W"))
            
            print('\t\tDS Train: {0} - {1} | Test : {2} - {3}'.format(train_begin.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"), test_begin.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")))

            all_paths[train_end.strftime("%Y%W")] = {
                'train_data_path' : path + 'train.csv',
                'eval_data_path' : path + 'eval.csv',
                'predict_data_path' : path + 'predict.csv',
                'dates_data_path' : path + 'dates.csv',
            }
            
            if lead_time_mode == 'w':
                train_begin = train_begin + relativedelta(weeks = lead_time)
            else:
                train_begin = train_begin + relativedelta(months = lead_time)

            train_end = train_begin + relativedelta(months = training_length)
            
            if lead_time_mode == 'w':
                prediction_month = train_end + relativedelta(weeks = lead_time)
            else:
                prediction_month = train_end + relativedelta(months = lead_time)
                
            test_begin = train_begin
            test_end = train_end + relativedelta(months = test_size)
            
        return all_paths