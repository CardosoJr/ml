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
from modelos.generic_model.ml_pipe.Tasks import backtest_generator
import modelos.generic_model.generic_tasks as tasks
import copy
import pickle

def run(config):
    
    def model_fn(parameters):
        print("################################################################################")
        print(parameters)
        print("################################################################################")

        selected_features = gridsearch_generator.get_features(parameters, config['variable_groups'])

        if len(selected_features) > 0:
            task_engineer_params = {
                'method' : config_methods.engineer_creator_name,
                'engineer' : { 'small_categorical' : config['small_categorical'],
                               'large_categorical' : config['large_categorical'],
                               'variable_set' : selected_features}
            }

            key = 'teste'
            task_model_params = {
                'method' : config_methods.model_creator_name,
                'model' : {'variable_set' : selected_features,
                           'log_experiment' : config['save_experiment'],
                           'model_params' : {'colsample_bytree': 0.75,
                                             'eta': 0.375,
                                             'gamma': 0.15000000000000002,
                                             'max_depth': 5,
                                             'min_child_weight': 7.0,
                                             'n_estimators': parameters['iterations'],
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
                            'elasticity_col' : config['elasticity_variables']}
            }

            task_metrics_params = {
                'method' : config_methods.metrics_creator_name,
                'method_regional' : '',
                'model_name' : config['model_name'],
                'score_params' : { 'training' : (config['initial_training_month'], 1) ,
                                   'oos' : (config['initial_training_month'], 1)}            
                }


            bkt_params = backtest_generator.BacktestParameters(
                    ds_name = config['ds_name'],
                    base_dir = config['base_dir'] + "{0}/".format(parameters['train_length']),
                    table_name = config['table_name'],
                    all_features = gridsearch_generator.get_all_possible_features(config['variable_groups']),
                    dataset_filter = config['dataset_filter'],
                    analysis_variables  = config['analysis_variables'],
                    train_eval_creator_name = config_methods.train_eval_creator_name,
                    predict_score_creator_name = config_methods.predict_score_creator_name,    
                    task_engineer_params = task_engineer_params,
                    task_model_params = task_model_params,
                    task_predict_params = task_predict_params,
                    task_metrics_params = task_metrics_params)


            parameter_generator = lambda: bkt_params.create_parameters(
                             initial_training_month = config['initial_training_month'], 
                             last_predicting_month = config['last_predicting_month'], 
                             lead_time = config['ld'], 
                             lead_time_mode = config['ld_mode'],
                             training_length = parameters['train_length'],
                             training_length_mode = config['tl_mode'],
                             test_length = config['testl'],
                             test_length_mode = config['testl_mode'],
                             stride_length = config['sl'],
                             stride_length_mode = config['sl_mode'])

            task_generator = lambda params: tasks.TaskModelMetrics(**params)

            backtest_tasks = backtest_generator.CreateTaskList(task_constructor  = task_generator, 
                                                               parameter_generator = parameter_generator)
            params =  parameter_generator()
            bkt_task = tasks.BacktestReport(task_bkt_params = params)
            d6tflow.run(bkt_task, workers = config['workers'])
            metric = gridsearch_generator.calculate_metric(bkt_task.output()['full_metrics'].load(), config['metric_weights'])
            gridsearch_generator.save_model_data(bkt_task.output()['full_metrics'].load(), parameters, metric, file_path = config['run_dir'])

        else:
            metric = np.inf

        print("################################################################################")
        print('Metric', metric)
        print("################################################################################")

        return metric
    
    
    d6tflow.set_dir(config['run_dir'] + 'data/')
    create_folder(config['run_dir'] + 'report/')
    
    k = config['k']
    with open(config['opt_file_name'], "rb") as f:
        tpe_trials = pickle.load(f)
    
    loss_par = list(set([(x['result']['loss'], str(x['misc']['vals'])) for x in tpe_trials.trials]))
    loss_par = sorted(loss_par, key = lambda tup: tup[0])
    pars = [x[1] for x in loss_par][:k]
    
    
    for par in pars: 
        par = json.loads(par.replace("'", "\""))
        for key in par.keys():
            par[key] = par[key][0]
            
            if 'group' in key:
                if par[key] == 1:
                    par[key] = False,
                else:
                    par[key] = True,
                    
            if 'iterations' in key or 'train_length' in key:
                par[key] = int(par[key])
                    
        print('running for par', par)
        model_fn(par)
        
    def group_dict(x):
        if x[0] == 0:
            return True
        else:
            return False

    def identity(x):
        return x[0]

    translation = tunning_fs.get_translation_dict(tunning_trials = tpe_trials,
                                                      identity_func = identity, 
                                                      group_func = group_dict)

    results = tunning_fs.parse_hyperopt_pickle(tunning_trials = tpe_trials, translation = translation)
        
    
    search = {'results' : results, 'group_translation' : config['variable_groups']}
    with open(config['run_dir'] + 'search.pkl', 'wb') as file:
        pickle.dump(search, file)
    