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

class OptParameters:
    '''
    description:
        Creates all opt parameters for tasks. 
        
        In this class we are fixing all paramters from all tasks except the Create_Train_Eval Task and Create_Predict_Score tests. In these tests, we are changing the range of period in each dataset and the directory where they're being saved. 
        
        UPDATE: Task Metrics are also updated. We change the parameters of date filters to calculate the metrics
        
        TODO: create a class in which all parameters may change, e.g. in some tests we're using a different set of vars. 
        
    input:
    
    output:
    
    '''
    def __init__(self, 
            ds_name = '',
            base_dir = '',
            table_name = '`sas-auto-marketprice-analytics.MKT_PRICING.BASE_MOD_RENOV_201801_201907_key_ajuste`',
            all_features = [],
            dataset_filter = [],
            analysis_variables  = [],
            train_eval_creator_name = '',
            predict_score_creator_name = '',    
            task_engineer_params = {},
            task_model_params = {},
            task_predict_params = {},
            task_metrics_params = {},
            task_el_ds_params= {},
            task_elasticity_params = {},
            task_opt_params = {},
            use_random_gpu = True):
        
        self.use_random_gpu = use_random_gpu
        self.ds_name = ds_name
        self.base_dir = base_dir
        self.table_name = table_name
        self.all_features = all_features
        self.dataset_filter = dataset_filter
        self.analysis_variables  = analysis_variables
        self.train_eval_creator_name =  train_eval_creator_name
        self.predict_score_creator_name = predict_score_creator_name
        
        self.task_engineer_params = task_engineer_params
        self.task_model_params = task_model_params
        self.task_predict_params = task_predict_params
        self.task_metrics_params = task_metrics_params
        self.task_el_ds_params = task_el_ds_params
        self.task_elasticity_params = task_elasticity_params    
        self.task_opt_params = task_opt_params
    
    
    def _update_opt_params(self, train_begin, train_end, elasticity_begin, elasticity_end):
        task_opt_params = self.task_opt_params
        task_opt_params['train_tag'] = train_begin.strftime("%Y%W") + "_" + train_end.strftime("%Y%W")
        task_opt_params['prediction_tag'] = elasticity_begin.strftime("%Y%W") + "_" + elasticity_end.strftime("%Y%W") 
        task_opt_params['prediction_begin'] = elasticity_begin
        task_opt_params['prediction_end'] = elasticity_end
        
        return task_opt_params
        
    def _update_elasticity_params(self, elasticity_begin, elasticity_end):
        task_el_ds_params = self.task_el_ds_params
        task_el_ds_params['elasticity_begin'] = elasticity_begin
        task_el_ds_params['elasticity_end'] = elasticity_end
        
    
        task_elasticity_params = self.task_elasticity_params
        return task_el_ds_params, task_elasticity_params
    
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
            pars['model']['model_params']['gpu_id'] = np.random.randint(2)
        return pars
        
    def _create_parameter(self, train_begin, train_end, test_begin, test_end):
        taks_te_params = {
            'method' : self.train_eval_creator_name,
            'create_ds' :  {  'ds_name' : self.ds_name, 
                              'dir' : self.base_dir,
                              'table_name' : self.table_name},
            'ds_params' :  {  'period_begin' : train_begin,
                              'period_end' : train_end,
                              'cols' : self.all_features,
                              'where' : self.dataset_filter},
            'key' : train_end.strftime("%Y%W")
        }

        task_ps_params = {
            'method' : self.predict_score_creator_name,
            'create_ds' : {  'ds_name' : self.ds_name, 
                             'dir' : self.base_dir,
                             'table_name' : self.table_name},
            'ds_params' : {  'period_begin' : test_begin,
                              'period_end' : test_end,
                              'cols' : self.all_features,
                              'analysis_variables' : self.analysis_variables,
                              'where' : self.dataset_filter},
            'key' : train_end.strftime("%Y%W") + "_" + test_end.strftime("%Y%W")
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
        
        train_end = OptParameters._add_time(train_begin, training_length, training_length_mode)
        prediction_period = OptParameters._add_time(train_end, lead_time, lead_time_mode)
        prediction_period_end = OptParameters._add_time(prediction_period, lead_time, lead_time_mode)
        test_begin = train_begin
        test_end = OptParameters._add_time(train_end, test_length, test_length_mode) 
        
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
        parameters = []
        
        while train_begin <= OptParameters._add_time(last_predicting_month, -2 * lead_time, lead_time_mode):
            train_end, prediction_period, prediction_period_end, test_begin, test_end = OptParameters._get_times(train_begin, 
                                                                                                                  lead_time,
                                                                                                                  lead_time_mode, 
                                                                                                                  training_length,
                                                                                                                  training_length_mode, 
                                                                                                                  test_length,
                                                                                                                  test_length_mode)
        
            train_begin_aux = train_begin
            train_end_aux = train_end
            while prediction_period <= last_predicting_month:
                taks_te_params, task_ps_params = self._create_parameter(train_begin, train_end, train_begin, test_end)
                metric_params_updated = self._update_metrics_params(train_begin, training_length, lead_time, prediction_period, lead_time)
                model_params_updated = self._update_model_params()
            
                task_el_ds_params, task_elasticity_params = self._update_elasticity_params(prediction_period, prediction_period_end)
                task_opt_params = self._update_opt_params( train_begin, train_end, prediction_period, prediction_period_end)
                
                parameters.append({
                    'taks_te_params'         : copy.deepcopy(taks_te_params), 
                    'task_ps_params'         : copy.deepcopy(task_ps_params), 
                    'task_engineer_params'   : self.task_engineer_params,
                    'task_model_params'      : copy.deepcopy(model_params_updated),
                    'task_predict_params'    : self.task_predict_params,
                    'task_metrics_params'    : copy.deepcopy(metric_params_updated), 
                    'task_el_ds_params'      : copy.deepcopy(task_el_ds_params),
                    'task_elasticity_params' : copy.deepcopy(task_elasticity_params),
                    'task_opt_params'        : copy.deepcopy(task_opt_params)})
            
                train_begin_aux = OptParameters._add_time(train_begin_aux, stride_length, stride_length_mode)
                train_end_aux, prediction_period, prediction_period_end, test_begin, test_end = OptParameters._get_times(train_begin_aux, 
                                                                                                                      lead_time,
                                                                                                                      lead_time_mode, 
                                                                                                                      training_length,
                                                                                                                      training_length_mode, 
                                                                                                                      test_length,
                                                                                                                      test_length_mode)
                
            train_begin = OptParameters._add_time(train_begin, stride_length, stride_length_mode)
                
        return parameters

