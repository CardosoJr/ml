import sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc
import json

import pandas as pd
from pathlib import Path
import os

from modelos.generic_model.ml_pipe.utils import create_folder
from modelos.generic_model.ml_pipe.Tasks import backtest_generator, gridsearch_generator
import modelos.generic_model.generic_tasks as tasks
import copy

import yaml
from datetime import datetime

from hyperopt.pyll.base import scope
from hyperopt import hp, tpe, fmin, Trials


def _create_hyperopt_pars(config_data):
    space = {}
    
    for key in config_data['params_to_opt'].keys():
        if config_data['opt_params'][key]['type'] == 'int_discrete':
            space[key] = scope.int(hp.quniform(key, 
                                               config_data['opt_params'][key]['first'], 
                                               config_data['opt_params'][key]['last'], 
                                               config_data['opt_params'][key]['step']))
            
        elif config_data['opt_params'][key]['type'] == 'choice':
            space[key] = hp.choice(key, config_data['opt_params'][key]['values'])
        
        elif config_data['opt_params'][key]['type'] == 'float_discrete':
            space[key] = hp.quniform(key, 
                                     config_data['opt_params'][key]['first'], 
                                     config_data['opt_params'][key]['last'], 
                                     config_data['opt_params'][key]['step'])
            
        elif config_data['opt_params'][key]['type'] == 'float':
            space[key] = hp.uniform(key, 
                                     config_data['opt_params'][key]['first'], 
                                     config_data['opt_params'][key]['last'])
            
        elif config_data['opt_params'][key]['type'] == "log_float":
            space[key] = hp.loguniform(key, 
                                     config_data['opt_params'][key]['first'], 
                                     config_data['opt_params'][key]['last'])
            
        else:
            raise Exception("Opt Param with invalid type {0}".format(config_data['opt_params'][key]['type']))
            
    if (config_data['opt_groups']):
        for group in config_data['variable_groups'].keys():
            space[group] = hp.choice(group, [True, False])
        
    return space

def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
        
    config_data['table_name'] = "`" + config_data['table_name'] + "`"
    config_data['initial_training_month'] = datetime.strptime(config_data['initial_training_month'],'%d-%m-%Y')
    config_data['last_predicting_month'] = datetime.strptime(config_data['last_predicting_month'],'%d-%m-%Y')
    config_data['space'] = _create_hyperopt_pars(config_data)
    return config_data

def run(config_path):  
    config = handle_config(config_path)
    # defining function to optimize
    def model_fn(parameters):
        print("################################################################################")
        print(parameters)
        print("################################################################################")

        config_cp = gridsearch_generator.update_config_tunning(copy.deepcopy(config), parameters)

        if len(config_cp['dataset_generator_params']['all_features']) > 0:
            bkt_params = backtest_generator.BacktestParameters(
                    config                = config_cp,
                    ds_name               = config_cp['ds_name'],
                    base_dir              = config_cp['base_dir'] + "{0}/".format(datetime.now().strftime("%Y%m%d%H%M%f")),
                    table_name            = config_cp['table_name'],
                    all_features          = gridsearch_generator.get_all_possible_features(config_cp['variable_groups']),
                    dataset_filter        = config_cp['dataset_filter'],
                    analysis_variables    = config_cp['analysis_variables'],
                    date_col              = config_cp['original_params']['date_col'])


            parameter_generator = lambda: bkt_params.create_parameters(
                             initial_training_month        = config_cp['initial_training_month'], 
                             last_predicting_month         = config_cp['last_predicting_month'], 
                             lead_time                     = config_cp['dataset_generator_params']['ld'], 
                             lead_time_mode                = config_cp['dataset_generator_params']['ld_mode'],
                             training_length               = config_cp['dataset_generator_params']['tl'],
                             training_length_mode          = config_cp['dataset_generator_params']['tl_mode'],
                             test_length                   = config_cp['dataset_generator_params']['testl'],
                             test_length_mode              = config_cp['dataset_generator_params']['testl_mode'],
                             stride_length                 = config_cp['dataset_generator_params']['sl'],
                             stride_length_mode            = config_cp['dataset_generator_params']['sl_mode'])

            task_generator = lambda params: tasks.TaskModelMetrics(**params)

            backtest_tasks = backtest_generator.CreateTaskList(task_constructor  = task_generator, 
                                                               parameter_generator = parameter_generator)
            params =  parameter_generator()
            bkt_task = tasks.BacktestReport(task_bkt_params = params)
            d6tflow.run(bkt_task, workers = config_cp['workers'])
            gridsearch_generator.calculate_metric(bkt_task.output()['full_metrics'].load(), config_cp['metric_weights'])

            gridsearch_generator.save_metrics(bkt_task.output()['full_metrics'].load(), parameters, file_path = config_cp['run_dir'])
            metric = gridsearch_generator.calculate_metric(bkt_task.output()['full_metrics'].load(), config_cp['metric_weights'])
        else:
            metric = np.inf

        print("################################################################################")
        print('Metric', metric)
        print("################################################################################")
    
        return metric
    
    
    d6tflow.set_dir(config['run_dir'] + 'data/')
    create_folder(config['run_dir'] + 'report/')
    
    
    opt_filename, best_pars = gridsearch_generator.run_opt(model_fn = model_fn, 
                                 parameter_space = config['space'], 
                                 max_iter = config['hyperopt_max_iterations'], 
                                 save_iter = config['hyperopt_save_iterations'],
                                 load = config['hyperopt_load'], 
                                 path = config['run_dir'])
    
    ## Plotting optimization report 
    report = GridSearchReport.Report(opt_file = opt_filename,
                 report_path = config['run_dir'] + 'report/' ,
                 test_id = config['opt_name'])
    
    report.plot_optimization_overview()
    report.plot_k_best(k = 10)
    
    ## Running best model 
    
    ## plotting backtesting report
