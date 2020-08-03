import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc
import threading
import pandas as pd

from modelos.generic_model.ml_pipe.utils import create_folder
from modelos.generic_model.ml_pipe.Tasks import backtest_generator
import modelos.generic_model.generic_tasks as tasks
import .generic_opt_tasks as opt_tasks

from datetime import datetime

"""
TODO: UPDATE FOR NEW FRAMEWORK FORMAT
"""

def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['table_name'] = "`" + config_data['table_name'] + "`"
    config_data['initial_training_month'] = datetime.strptime(config_data['initial_training_month'],'%d-%m-%Y')
    config_data['last_predicting_month'] = datetime.strptime(config_data['last_predicting_month'],'%d-%m-%Y')
    config_data['feature_label'] = config_data['all_features'] + [config_data['label']]
    return config_data

def run(config_path):      
    config = handle_config(config_path)
    d6tflow.set_dir(config['run_dir'] + 'data/')
    
    task_engineer_params = {
        'method' : config_methods.engineer_creator_name,
        'engineer' : {'small_categorical' : config['small_categorical'],
                                  'large_categorical' : config['large_categorical'],
                                  'variable_set' : config['all_features']}
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
                                     'random_seed' : 42}},
        'build_params' : {'name' : key}
    }
    
    task_predict_params = {
        'predict_params' : {
                    'elasticity_transform' : False,
                    'elasticity_factor' : 1,
                    'elasticity_col' : []}
    }
    
    task_metrics_params = {
        'method' : config_methods.metrics_creator_name,
        'model_name' : config['model_name'],
        'method_regional' : '',
        'score_params' : None            
        }
    
    task_el_ds_params = {
        'date_col' : config['original_params']['date_col'],
        'elasticity_begin' : None,
        'elasticity_end' : None,
        'n_min' : config['factor_min'],
        'n_max' : config['factor_max'],
        'qtd_pass' : config['num_points'],
    }
    
    task_elasticity_params = {
        'model_name' : config['model_name'],
        'predict_params' : {
                    'elasticity_transform' : False,
                    'elasticity_factor' : 1,
                    'elasticity_col' : 'TESTE'}
    } 
    
    task_opt_params = {
        'train_tag' : None,
        'prediction_tag' : None ,
        'prediction_begin' : None,
        'prediction_end' : None,
        'target' : config['params']['target'],
        'output' : config['params']['model_output'],
        'key' : config['original_params']['hash_col'],
        'date_col': config['params']['date_col'],
        'metrics' : ['perc_err', 'auc', 'elasticity_err'],
     }
    
    opt_params = optm_generator.OptParameters(
            ds_name = config['ds_name'],
            base_dir = config['base_dir'],
            table_name = config['table_name'],
            all_features = config['all_features'],
            dataset_filter = config['dataset_filter'],
            analysis_variables  = config['analysis_variables'],
            train_eval_creator_name = config_methods.train_eval_creator_name,
            predict_score_creator_name = config_methods.predict_score_creator_name,    
            task_engineer_params = task_engineer_params,
            task_model_params = task_model_params,
            task_predict_params = task_predict_params,
            task_metrics_params = task_metrics_params,
            task_el_ds_params = task_el_ds_params, 
            task_elasticity_params = task_elasticity_params,
            task_opt_params = task_opt_params)
       
    params = opt_params.create_parameters(
                     initial_training_month = config['initial_training_month'], 
                     last_predicting_month = config['last_predicting_month'], 
                     lead_time = config['ld'], 
                     lead_time_mode = config['ld_mode'],
                     training_length = config['tl'],
                     training_length_mode = config['tl_mode'],
                     test_length = config['testl'],
                     test_length_mode = config['testl_mode'],
                     stride_length = config['sl'],
                     stride_length_mode = config['sl_mode'])
    
    print('NUM_OPT', len(params))
    
    to_run_task = opt_tasks.TaskOptSummary(task_opt_summary_params = params)
    d6tflow.run(to_run_task, workers = config['workers'])
    
    create_folder(config['run_dir'] + 'report/')
    
    result = to_run_task.output().load()
    result.to_csv('opt_result.csv', index = False)