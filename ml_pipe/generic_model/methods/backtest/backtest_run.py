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
from generic_model.ml_pipe.Tasks import backtest_generator
import generic_model.generic_tasks as tasks

from datetime import datetime

import yaml

def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['table_name'] = "`" + config_data['table_name'] + "`"
    config_data['initial_training_month'] = datetime.strptime(config_data['initial_training_month'],'%d-%m-%Y')
    config_data['last_predicting_month'] = datetime.strptime(config_data['last_predicting_month'],'%d-%m-%Y')
    config_data['feature_label'] = config_data['dataset_generator_params']['all_features'] + [config_data['label']]
    
    return config_data


def run(config_path):     
    config = handle_config(config_path)
    
    d6tflow.set_dir(config['run_dir'] + 'data/')
    
    bkt_params = backtest_generator.BacktestParameters(
            config                 = config,
            ds_name                = config['ds_name'],
            base_dir               = config['base_dir'],
            table_name             = config['table_name'],
            all_features           = config['dataset_generator_params']['all_features'],
            dataset_filter         = config['dataset_filter'],
            analysis_variables     = config['analysis_variables'],
            date_col               = config['original_params']['date_col'])
    
    
    parameter_generator = lambda: bkt_params.create_parameters(
                     initial_training_month = config['initial_training_month'], 
                     last_predicting_month  = config['last_predicting_month'], 
                     lead_time              = config['dataset_generator_params']['ld'], 
                     lead_time_mode         = config['dataset_generator_params']['ld_mode'],
                     training_length        = config['dataset_generator_params']['tl'],
                     training_length_mode   = config['dataset_generator_params']['tl_mode'],
                     test_length            = config['dataset_generator_params']['testl'],
                     test_length_mode       = config['dataset_generator_params']['testl_mode'],
                     stride_length          = config['dataset_generator_params']['sl'],
                     stride_length_mode     = config['dataset_generator_params']['sl_mode'])
    
    task_generator = lambda params: tasks.TaskModelMetrics(**params)
    
    backtest_tasks = backtest_generator.CreateTaskList(task_constructor  = task_generator, 
                                                       parameter_generator = parameter_generator)
    params =  parameter_generator()
    bkt_task = tasks.BacktestReport(task_bkt_params = params)
    d6tflow.run(bkt_task, workers = config['workers'])
    
    create_folder(config['run_dir'] + 'report/')
    
    metrics_df = bkt_task.output()['full_metrics'].load()
    
    # Salvando o .csv para o frontend
    path_final = config['run_dir'] + 'df_backtest_final.csv'
    
    metrics_df.to_csv(path_final,index=False)