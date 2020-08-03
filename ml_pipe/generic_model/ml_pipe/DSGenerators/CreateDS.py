import sys
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import gc
pd.set_option('display.max_columns', 500)

from datetime import datetime
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

from generic_model.ml_pipe.utils import get_blob, read_multiple_csv
from google.cloud import storage
from datetime import datetime
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime
from datetime import timedelta  
from dateutil.relativedelta import relativedelta

class CreateDataSet:
    def __init__(self, 
                 seed = 9999,
                 run = True, 
                 project = 'sas-auto-marketprice-analytics', 
                 dataset = 'MKT_PRICING',
                 table_name = '`sas-auto-marketprice-analytics.MKT_PRICING.BASE_MOD_RENOV_201801_201907_key_ajuste`',
                 ds_name = 'teste', 
                 dir = ''):
        self.ds_name = ds_name
        self.run = run
        self.seed = seed
        self.project = project
        self.dataset = dataset
        self.table_name = table_name
        self.dir = dir
    
    def get_from_bq(self, output_table, sql, file_names):
        bq = bigquery.Client(project= self.project)
        # Configures the job that will export the query to a temporary table. This table will be exported to csv
        job_config = bigquery.QueryJobConfig()
        table_ref = bq.dataset(self.dataset).table(output_table)
        job_config.destination = table_ref
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

        query_job = bq.query(sql, job_config=job_config)
        query_job.result() 
        print("Query result loaded to: {}".format(table_ref.path))
        
        # Configure the job that will export the query result to csv files in the bucket
        dataset_ref = bq.dataset(self.dataset, project = self.project)
        table_ref = dataset_ref.table(output_table)
        
        extract_job = bq.extract_table(table_ref, file_names) 
        extract_job.result()

        # Deletes the temporary table
        print('Query results extracted to GCS: {}'.format(file_names))
        bq.delete_table(table_ref) #Deletes table in BQ
        print('Table {} deleted'.format(table_ref))
    
    
    def get_path(self, dir = ''):
        train_path = dir + 'train.csv'
        eval_path = dir + 'eval.csv'
        predict_path = dir + 'predict.csv'
        dates_path = dir + 'dates.csv'
        return {
            'train_data_path' : train_path,
            'eval_data_path' : eval_path,
            'predict_data_path' : predict_path,
            'dates_data_path' : dates_path
        }
    
    def set_run(self, run = True):
        self.run = run
    
    def create_train_eval_bq(self, 
                          period_begin,
                          period_end,
                          date_col = 'DAT_FIM_VIGENCIA',
                          cols = [],
                          target = 'QTD_PROP',
                          where = '',
                          train_size = 0.8,
                          key_col = 'key'):
        
        output_table = datetime.now().strftime("%Y%m%d%H%M%f") + '_aux'
        query = "SELECT {0} FROM {1} ".format(','.join(cols + [target] + [key_col]), self.table_name)
        
        original_query = query + " " + where + " AND "

        sql = original_query + """ 
            CAST({0} AS DATE) >= CAST(TIMESTAMP '{1}' AS DATE) AND
            CAST({0} AS DATE) < CAST(TIMESTAMP '{2}' AS DATE)
        """.format(date_col, period_begin.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"))

        master_path = self.dir + "{0}/{1}/".format(self.ds_name, period_begin.strftime("%Y%W") + "_" + period_end.strftime("%Y%W")) + 'master*.csv'
        print(master_path)
        self.get_from_bq(output_table, sql, master_path)
        print('\t\tTrain - Test Splitting')
        test_size = 1 - train_size
        master = read_multiple_csv('marketprice', self.dir + "{0}/{1}/".format(self.ds_name, period_begin.strftime("%Y%W") + "_" + period_end.strftime("%Y%W")) + "master")
        train_df, eval_df = train_test_split(master, test_size= test_size, random_state = self.seed)
        
        train_keys = pd.DataFrame([])
        train_keys[key_col] = train_df[key_col]
        train_keys['MODEL_TYPE'] = ['TRAIN'] * len(train_keys)
        
        eval_keys = pd.DataFrame([])
        eval_keys[key_col] = eval_df[key_col]
        eval_keys['MODEL_TYPE'] = ['EVAL'] * len(eval_keys)
        
        train_eval_keys = train_keys.append(eval_keys, ignore_index = True)
        
        
        train_df = train_df[cols + [target]]
        eval_df = eval_df[cols + [target]]
        
        del train_keys
        del eval_keys
        del master
        gc.collect()
        
        return train_df, eval_df, train_eval_keys
        
    def __hash__(self):
        return hash(repr(self))
    
    def create_predict_scoring_bq(self, 
                          period_begin,
                          period_end,
                          date_col = 'DAT_FIM_VIGENCIA',
                          target = 'QTD_PROP',
                          where = '',
                          cols = [],
                          key_col = 'key',        
                          analysis_variables =  ['key', 'CATEGORIA', 'COD_TIPO_RENOV', 'REGIONAL', 'PREM_EMITIDO_ATU', 'PREM_EMITIDO_ANUAL']):
        
        output_table = datetime.now().strftime("%Y%m%d%H%M%f") + '_aux'
        date_columns = list([date_col])
        date_columns.extend(analysis_variables)
        not_found = [x for x in date_columns if x not in cols]

        all_cols = cols + [target] +  not_found

        predict_query = """ SELECT {0} FROM {1} {2} AND CAST({3} AS DATE) >= CAST(TIMESTAMP '{4}' AS DATE) AND CAST({3} AS DATE) < CAST(TIMESTAMP '{5}' AS DATE) """.format(
            ','.join(all_cols), 
            self.table_name, 
            where,
            date_col, 
            period_begin.strftime("%Y-%m-%d"),
            period_end.strftime("%Y-%m-%d"))

        test_path = self.dir + "{0}/{1}/".format(self.ds_name, period_begin.strftime("%Y%W") + "_" + period_end.strftime("%Y%W")) + 'full_predict_dates*.csv'
        self.get_from_bq(output_table, predict_query, test_path)

        print('\t\tSplitting Predict / Dates')
        predict_dates = read_multiple_csv('marketprice', self.dir + "{0}/{1}/".format(self.ds_name, period_begin.strftime("%Y%W") + "_" + period_end.strftime("%Y%W")) + "full_predict_dat")
        dates_df = predict_dates[date_columns + [target]]
        predict_df = predict_dates.drop(columns = not_found)
        
        del predict_dates
        gc.collect()
        
        return predict_df, dates_df