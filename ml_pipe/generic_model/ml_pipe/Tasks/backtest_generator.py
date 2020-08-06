import pandas as pd
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from pathlib import Path
import os
    
import d6tflow
import luigi
import gc
from dateutil.relativedelta import relativedelta
import copy

class BacktestParameters:
    '''
    description:
        Creates all backtest parameters for tasks. 
        
        In this class we are fixing all paramters from all tasks except the Create_Train_Eval Task and Create_Predict_Score tests. In these tests, we are changing the range of period in each dataset and the directory where they're being saved. 
        
        UPDATE: Task Metrics are also updated. We change the parameters of date filters to calculate the metrics
        
        TODO: create a class in which all parameters may change, e.g. in some tests we're using a different set of vars. 
        
    input:
    
    output:
    
    '''
    def __init__(self, 
            config,
            ds_name = '',
            base_dir = '',
            table_name = '`sas-auto-marketprice-analytics.MKT_PRICING.BASE_MOD_RENOV_201801_201907_key_ajuste`',
            all_features = [],
            dataset_filter = [],
            analysis_variables  = [],
            date_col='',
            use_random_gpu = True,
            available_gpu = [0, 1]):
        
        self.use_random_gpu = use_random_gpu
        self.ds_name = ds_name
        self.base_dir = base_dir
        self.table_name = table_name
        self.all_features = all_features
        self.dataset_filter = dataset_filter
        self.analysis_variables  = analysis_variables
        self.date_col = date_col
        self.available_gpu = available_gpu
        
        self.config = config
        
        self.task_engineer_params = {
            'method' : list(config['task_engineer_params']['methods'].keys()),
            'engineer' : BacktestParameters.get_engineer_params(config)}

        key = 'teste'
        self.task_model_params = {
            'method' : config['task_model_params']['method'],
            'model' : {'model_params'   : config['task_model_params']['model_params'],
                       'ds_params': {'date_col': config['original_params']['date_col'],
                                     'target': config['original_params']['target'],
                                     'hash_col': config['original_params']['hash_col']}},

            'build_params' : {'name' : key}}

        self.task_predict_params = {
            'predict_params' : {
                        'elasticity_transform' : False,
                        'elasticity_factor' : 1,
                        'elasticity_col' : config['elasticity_variables']}}

        self.task_metrics_params = {
            'method'          : list(config['task_metric_params']['methods'].keys()),
            'metrics'         : config['task_metric_params']['methods'],
            'model_name'      : config['model_name'],
            'score_params'    : { 'training' : (config['initial_training_month'], 1) ,
                                  'oos' : (config['initial_training_month'], 1)}}        
    
    @staticmethod
    def get_engineer_params(config):
        eng_params = {}
        
        for method, params in config['task_engineer_params']['methods'].items():
            eng_params[method] = {
                **params,
                'variable_set':       config['dataset_generator_params']['all_features'],
                'date':               config['original_params']['date_col'],
                'target':             config['original_params']['target']
            }
        
        return eng_params
    
    def _update_metrics_params(self, train_begin, train_length, lead_time_length, prediction_begin, prediction_length):
        pars = copy.deepcopy(self.task_metrics_params)
        pars['score_params'] = {'erro_train' : (train_begin, train_length),
#                                 'lead_time' : (train_end, lead_time_length), 
                                 'erro_oos' : (prediction_begin, prediction_length)}
        
        return pars
    
    def _update_metrics_params_old(self, train_begin, train_end, test_begin, test_end, prediction_begin, prediction_end):
        pars = copy.deepcopy(self.task_metrics_params)
        pars['score_params'] = {'erro_train' : (train_begin, train_end),
                                'lead_time' : (train_end, prediction_begin), 
                                 'erro_oos' : (prediction_begin, prediction_end)}
        
        return pars
    
    def _update_model_params(self):
        pars = copy.deepcopy(self.task_model_params)
        if self.use_random_gpu:
            pars['model']['model_params']['gpu_id'] = self.available_gpu[np.random.randint(len(self.available_gpu))]
        return pars
        
    def _create_parameter(self, train_begin, train_end, test_begin, test_end):
        taks_te_params = {
            'method' : self.config['dataset_generator_params']['method'],
            'create_ds' :  {  'ds_name'      : self.ds_name, 
                              'dir'          : self.base_dir,
                              'table_name'   : self.table_name,
                              'project'      : self.table_name.replace("`", "").split(".")[0],
                              'dataset'      : self.table_name.replace("`", "").split(".")[1]},
            'ds_params' :  {  'period_begin' : train_begin,
                              'period_end'   : train_end,
                              'cols'         : self.all_features,
                              'where'        : self.dataset_filter,
                              'date_col'     : self.date_col,
                              'target'       : self.config['original_params']['target']},
            'key' : train_end.strftime("%Y%W")
        }

        task_ps_params = {
            'method' : self.config['dataset_generator_params']['method'],
            'create_ds' : {  'ds_name'      : self.ds_name, 
                             'dir'          : self.base_dir,
                             'table_name'   : self.table_name,
                             'project'      : self.table_name.replace("`", "").split(".")[0],
                             'dataset'      : self.table_name.replace("`", "").split(".")[1]},
            'ds_params' : {  'period_begin' : test_begin,
                              'period_end' : test_end,
                              'cols' : self.all_features,
                              'analysis_variables' : self.analysis_variables,
                              'where' : self.dataset_filter,
                              'date_col': self.date_col,
                              'target'       : self.config['original_params']['target']},
            'key' : train_end.strftime("%Y%W")
        }
        
        return taks_te_params, task_ps_params
    
    @staticmethod
    def _add_time(base_dt, add_value, add_mode):
        if add_mode == 'm':
            base_dt = base_dt + relativedelta(months = add_value)
        else:
            base_dt = base_dt + relativedelta(weeks = add_value)
        return base_dt
    
    @staticmethod
    def _get_times(train_begin, 
                     lead_time = 1, 
                     lead_time_mode = 'm',
                     training_length = 3,
                     training_length_mode = 'm',
                     test_length = 3,
                     test_length_mode = 'm'):
        
        train_end = BacktestParameters._add_time(train_begin, training_length, training_length_mode)
        prediction_period = BacktestParameters._add_time(train_end, lead_time, lead_time_mode)
        prediction_period_end = BacktestParameters._add_time(prediction_period, lead_time, lead_time_mode)
        test_begin = train_begin
        test_end = BacktestParameters._add_time(train_end, test_length, test_length_mode) 
        
        return train_end, prediction_period, prediction_period_end, test_begin, test_end
    
        
    def create_parameters(self,
                     initial_training_month, 
                     last_predicting_month, 
                     lead_time = 1, 
                     lead_time_mode = 'm',
                     training_length = 3,
                     training_length_mode = 'm',
                     test_length = 3,
                     test_length_mode = 'm',
                     stride_length = 1,
                     stride_length_mode = 'm'):
        
        all_paths = {}
        train_begin = initial_training_month
        
        train_end, prediction_period, prediction_period_end, test_begin, test_end = BacktestParameters._get_times(train_begin, 
                                                                                                                  lead_time,
                                                                                                                  lead_time_mode, 
                                                                                                                  training_length,
                                                                                                                  training_length_mode, 
                                                                                                                  test_length,
                                                                                                                  test_length_mode)
        
        parameters = []
        while prediction_period <= last_predicting_month:
            taks_te_params, task_ps_params = self._create_parameter(train_begin, train_end, test_begin, test_end)
#             metric_params_updated = self._update_metrics_params(train_begin, train_end, test_begin, test_end, prediction_period, prediction_period_end)
            metric_params_updated = self._update_metrics_params(train_begin, training_length, lead_time, prediction_period, lead_time)
            model_params_updated = self._update_model_params()
            
            parameters.append({
                'taks_te_params'       : taks_te_params, 
                'task_ps_params'       : task_ps_params, 
                'task_engineer_params' : self.task_engineer_params,
                'task_model_params'    : model_params_updated,
                'task_predict_params'  : self.task_predict_params,
                'task_metrics_params'  : metric_params_updated})
            
            
            train_begin = BacktestParameters._add_time(train_begin, stride_length, stride_length_mode)
            train_end, prediction_period, prediction_period_end, test_begin, test_end = BacktestParameters._get_times(train_begin, 
                                                                                                                      lead_time,
                                                                                                                      lead_time_mode, 
                                                                                                                      training_length,
                                                                                                                      training_length_mode, 
                                                                                                                      test_length,
                                                                                                                      test_length_mode)
        return parameters

    
class CreateTaskList:
    def __init__(self, task_constructor, parameter_generator):
        self.task_constructor = task_constructor 
        self.parameter_generator = parameter_generator
    
    def get_tasks(self):
        tasks = []
        for par in self.parameter_generator():
            tasks.append(self.task_constructor(par))
        return tasks