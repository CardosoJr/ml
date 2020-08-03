import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc

import pandas as pd
from pathlib import Path
import os

from generic_model.ml_pipe.utils import create_folder
from generic_model.ml_pipe.Tasks import backtest_generator
import generic_model.generic_tasks as tasks
import .elasticity_tasks as el_tasks

import yaml
from datetime import datetime

"""
TODO: UPDATE FOR NEW FRAMEWORK!

"""



def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['table_name'] = "`" + config_data['table_name'] + "`"
    config_data['train_begin'] = datetime.strptime(config_data['train_begin'],'%d-%m-%Y')
    config_data['train_end'] = datetime.strptime(config_data['train_end'],'%d-%m-%Y')
    config_data['test_begin'] = datetime.strptime(config_data['test_begin'],'%d-%m-%Y')
    config_data['test_end'] = datetime.strptime(config_data['test_end'],'%d-%m-%Y')
    config_data['elasticity_begin'] = datetime.strptime(config_data['elasticity_begin'],'%d-%m-%Y')
    config_data['elasticity_end'] = datetime.strptime(config_data['elasticity_end'],'%d-%m-%Y')
    return config_data

def run(config_path):
    config = handle_config(config_path)
    
    d6tflow.set_dir(config['run_dir'] + 'data/')
    
    ## Deal with 
    taks_te_params = {
        'method' : config_methods.train_eval_creator_name,
        'create_ds' :    {    'ds_name' : config['ds_name'],
                              'dir' : config['base_dir'],
                              'table_name' : config['table_name']},
        'ds_params' :    {  'period_begin' : config['train_begin'],
                            'period_end' : config['train_end'],
                            'cols' : config['all_features'],
                            'where' : config['dataset_filter'],
                            'date_col': config['original_params']['date_col']},
        'key' : config['train_end'].strftime("%Y%W")
    }
    
    task_ps_params = {
        'method' : config_methods.predict_score_creator_name,
        'create_ds' : {  'ds_name' : config['ds_name'], 
                         'dir' : config['base_dir'],
                         'table_name' : config['table_name']},
        'ds_params' : {  'period_begin' : config['test_begin'],
                          'period_end' : config['test_end'],
                          'cols' : config['all_features'],
                          'analysis_variables' : config['analysis_variables'],
                          'where' : config['dataset_filter'],
                          'date_col': config['original_params']['date_col']},
        'key' : config['train_end'].strftime("%Y%W")
    }
    
    
    task_engineer_params = {
        'method' : config_methods.engineer_creator_name,
        'engineer' : {'small_categorical' : config['small_categorical'],
                                  'large_categorical' : config['large_categorical'],
                                  'variable_set' : config['all_features'],
                                  'date'         : config['original_params']['date_col']}
    }
    
    key = 'teste'
    task_model_params = {
        'method' : config_methods.model_creator_name,
        'model' : {
                   'model_params' : {'colsample_bytree': 0.75,
                                     'eta': 0.375,
                                     'gamma': 0.15000000000000002,
                                     'max_depth': 5,
                                     'min_child_weight': 7.0,
                                     'n_estimators': config['iterations'],
                                     'reg_alpha': 2.5500000000000003,
                                     'reg_lambda ': 0.12,
                                     'subsample': 0.75,
                                     'verbose' : 20,
                                     'eval_metric': 'logloss', # 'auc',
                                     'objective': 'binary:logistic',
                                     'booster': "gbtree",
                                     'tree_method':'gpu_hist',
                                     'gpu_id' : 0,
                                     'random_seed' : 42},
                    'ds_params': {'date_col': config['original_params']['date_col'],
                                 'target': config['original_params']['target'],
                                 'hash_col': config['original_params']['hash_col']}},
        'build_params' : {'name' : key}
    }
    
    
    task_el_ds_params = {
        'date_col' : config['original_params']['date_col'],
        'elasticity_begin' : config['elasticity_begin'],
        'elasticity_end' : config['elasticity_end'],
        'n_min' : config['factor_min'],
        'n_max' : config['factor_max'],
        'qtd_pass' : config['num_points'],
    }
    
    task_elasticity_params = {
        'model_name' : key,
        'predict_params' : {
                    'elasticity_transform' : False,
                    'elasticity_factor' : 1,
                    'elasticity_col' : 'TESTE'}
    }
    
    task_elasticity_report_params = {
        'model_name' : key,
        'n_min' : config['factor_min'],
        'n_max' : config['factor_max'], 
        'real_qtd_pass' : config['real_num_points'],
        'qtd_pass' : config['num_points'],
        'target' : config['original_params']['target'],
        'output' : config['params']['model_output']
    }
    
    t = el_tasks.TaskElasticityReport(
            taks_te_params = taks_te_params,
            task_ps_params = task_ps_params,
            task_engineer_params = task_engineer_params,
            task_model_params = task_model_params,
            task_el_ds_params = task_el_ds_params,
            task_elasticity_params = task_elasticity_params, 
            task_elasticity_report_params = task_elasticity_report_params)
    
    d6tflow.run(t, workers = config['workers'])
    
    real_df = t.output()['real_df'].load()
    model_df = t.output()['model_df'].load()    
    
    real_df.to_csv(config['run_dir'] + 'real_elasticity.csv', index = False)
    model_df.to_csv(config['run_dir'] + 'model_elasticity.csv', index = False)