import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc

import pandas as pd
import os
from pathlib import Path

from generic_model.ml_pipe.utils import create_folder
from generic_model.ml_pipe.Tasks.backtest_generator import BacktestParameters
import generic_model.generic_tasks as tasks

from datetime import datetime

import yaml

def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['table_name'] = "`" + config_data['table_name'] + "`"
    config_data['initial_training_month'] = datetime.strptime(config_data['initial_training_month'],'%d-%m-%Y')
    config_data['feature_label'] = config_data['dataset_generator_params']['all_features'] + [config_data['label']]
    
    return config_data


def run(config_path):     
    config = handle_config(config_path)
    d6tflow.set_dir(config['run_dir'] + 'data/')
    available_gpu = [0, 1] ## TODO USE AS PARAMETER
    
    train_end, prediction_period, prediction_period_end, test_begin, test_end = BacktestParameters._get_times(config['initial_training_month'], 
                                                                                                                  config['dataset_generator_params']['ld'],
                                                                                                                  config['dataset_generator_params']['ld_mode'], 
                                                                                                                  config['dataset_generator_params']['tl'],
                                                                                                                  config['dataset_generator_params']['tl_mode'], 
                                                                                                                  config['dataset_generator_params']['testl'],
                                                                                                                  config['dataset_generator_params']['testl_mode'])
    
    taks_te_params = {
            'method' : self.config['dataset_generator_params']['method'],
            'create_ds' :  {  'ds_name'      : config['ds_name'], 
                              'dir'          : config['base_dir'],
                              'table_name'   : config['table_name']},
            'ds_params' :  {  'period_begin' : config['initial_training_month'],
                              'period_end'   : train_end,
                              'cols'         : config['dataset_generator_params']['all_features'],
                              'where'        : config['dataset_filter'],
                              'date_col'     : config['original_params']['date_col'],
                              'target'       : config['original_params']['target']},
            'key' : train_end.strftime("%Y%W")
        }

    task_ps_params = {
            'method' : self.config['dataset_generator_params']['method'],
            'create_ds' : {  'ds_name'             : config['ds_name'],, 
                             'dir'                 : config['base_dir'],
                             'table_name'          : config['table_name']},
            'ds_params' : {  'period_begin'        : test_begin,
                              'period_end'         : test_end,
                              'cols'               : config['dataset_generator_params']['all_features'],
                              'analysis_variables' : config['analysis_variables'],
                              'where'              : config['dataset_filter'],
                              'date_col'           : config['original_params']['date_col'],
                              'target'             : config['original_params']['target']},
            'key' : train_end.strftime("%Y%W")
        }
    
    
    
    task_engineer_params = {
        'method' : list(config['task_engineer_params']['methods'].keys()),
        'engineer' : BacktestParameters.get_engineer_params(config)}

    task_model_params = {
        'method' : config['task_model_params']['method'],
        'model' : {'model_params'  : config['task_model_params']['model_params'],
                   'ds_params': {'date_col': config['original_params']['date_col'],
                                 'target': config['original_params']['target'],
                                 'hash_col': config['original_params']['hash_col']}},

        'build_params' : {'name' : 'single_run'}}
    
    task_model_params['model']['model_params']['gpu_id'] = available_gpu[np.random.randint(len(available_gpu))]

    task_predict_params = {
        'predict_params' : {
                    'elasticity_transform' : False,
                    'elasticity_factor' : 1,
                    'elasticity_col' : config['elasticity_variables']}}

    task_metrics_params = {
        'method' : 'standard',
        'model_name' : config['model_name'],
        'method_regional' : '',
        'score_params' : 'erro_train' : (config['initial_training_month'], train_end),
                         'erro_oos'   : (prediction_begin, prediction_end)
    }  
    
    parameters = {
                'taks_te_params'       : taks_te_params, 
                'task_ps_params'       : task_ps_params, 
                'task_engineer_params' : self.task_engineer_params,
                'task_model_params'    : model_params_updated,
                'task_predict_params'  : self.task_predict_params,
                'task_metrics_params'  : metric_params_updated}
    
    
    tk_metrics = tasks.TaskModelMetrics(**parameters)
    
    d6tflow.run(tk_metrics, workers = config['workers'])
    
    create_folder(config['run_dir'] + 'report/')
    
    metrics_df = tk_metrics.output()['full_metrics'].load()
    metrics_df.to_csv(config['run_dir'] + 'base_model.csv', index=False)