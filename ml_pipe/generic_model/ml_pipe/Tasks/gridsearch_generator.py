import pandas as pd
import numpy as np
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import d6tflow
import luigi
import gc
import os
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.pyll.base import scope
import pickle
import copy
from functools import reduce  # forward compatibility for Python 3
import operator

def __getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def __setInDict(dataDict, mapList, value):
    __getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def run_opt(model_fn, parameter_space, max_iter=10000, save_iter=100, load = False, name = 'opt', path = ''):
    '''
    Description
        runs hyperopt optimization
    
    Input:
        model_fn : model function, should return the value of the metric to be minimized 
        parameter_space : hyperopt parameter space to be optimized 
        max_iter : optmization max iterations
        save_iter : the results will be saved every save_iter iterations 
        load : loads previous saved results
        name : name of the optimization

    Output:
        best parameters

    '''
    file_name = path + "tpe_trials_{0}.p".format(name)
    i = 1
    max_i = round(max_iter/save_iter)
    if load and os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            tpe_trials = pickle.load(f)
    else:
        tpe_trials = Trials()
    while i <= max_i:
        print(parameter_space)
        best = fmin(fn =  model_fn, space = parameter_space, algo = tpe.suggest, trials = tpe_trials, max_evals = i * save_iter)
        with open(file_name, "wb") as f:
            pickle.dump(tpe_trials, f)
        i += 1
        gc.collect()
        
    return file_name, best

def save_model_data(model_data, opt_params, obj_value, file_path = ''): 	
    model_data['params'] = str(opt_params).replace(',', '\n').replace('[', '').replace(']', '').replace('.0', '')
    model_data['obj'] = obj_value
    
    if os.path.exists(file_path + "k_best_data.csv"):
        with open(file_path + 'k_best_data.csv', 'a') as f:
                model_data.to_csv(f, mode = 'a', header = False)
    else:
        model_data.to_csv(file_path + 'k_best_data.csv')


def save_metrics(full_metrics, opt_params, file_path = ''):
    params = copy.deepcopy(opt_params)
    
    df_el_err = full_metrics[(full_metrics['period'] == 'erro_oos') & (full_metrics['metric'] == 'elasticity_err')]
    
    # Elasticity Error Component
    params['mean_el_err'] = df_el_err['value'].mean()
    params['std_el_err'] = df_el_err['value'].std()
    params['median_el_err'] = df_el_err['value'].median()
    
    df_auc = full_metrics[(full_metrics['period'] == 'erro_oos') & (full_metrics['metric'] == 'auc')]
    
    # Auc Component
    df_auc['1-value'] = 1 - df_auc['value']
    params['mean_auc'] = df_auc['1-value'].mean() 
    params['std_auc'] = df_auc['1-value'].std()
    params['median_auc'] = df_auc['1-value'].median()
    
    df_err = full_metrics[(full_metrics['period'] == 'erro_oos') & (full_metrics['metric'] == 'perc_err')]
    
    params['mean_err'] = df_err['value'].mean()
    params['std_err'] = df_err['value'].std()
    params['median_err'] = df_err['value'].median()
    
    df = pd.DataFrame([params])
    
    if os.path.exists(file_path + "saved_results_opt.csv"):
        with open(file_path + 'saved_results_opt.csv', 'a') as f:
            df.to_csv(f, mode = 'a', header = False)
    else:
        df.to_csv(file_path + 'saved_results_opt.csv')
    

def calculate_metric(full_metrics, params):
    '''
    Description:
        Calculates the metric to be optimized 
    
    Input:
        df with all model metrics
        params : weight information
        
    Output:
        Value to be minimized
    '''
    df_el_err = full_metrics[(full_metrics['period'] == 'erro_oos') & (full_metrics['metric'] == 'elasticity_err')]
    
    # Elasticity Error Component
    metric = df_el_err['value'].mean() * params['w_el_err_mean'] + df_el_err['value'].std() * params['w_el_err_std']
    
    df_auc = full_metrics[(full_metrics['period'] == 'erro_oos') & (full_metrics['metric'] == 'auc')]
    
    # Auc Component
    df_auc['1-value'] = 1 - df_auc['value']
    metric = metric + df_auc['1-value'].mean() * params['w_auc_mean'] + df_auc['1-value'].std() * params['w_auc_std']
    
    return metric


def get_all_possible_features(variable_groups):
    '''
    Description
        gets all possible features in all groups
        
    Input:
        dictionary with all variable groups
        
    Output:
        List with all features
        
    '''
    aux = [np.array(variable_groups[x]) for x in variable_groups.keys()]
    all_variables = []
    _ = [all_variables.extend(x) for x in aux]
    all_possible_features = np.unique(all_variables)
    return sorted(list(all_possible_features))

def get_features(parameters, variable_sets):
    '''
    Description:
        gets all selected variables given the choosen parameters
    Input:
    Output:
    
    '''
    selected_features = []
    for group in variable_sets.keys():
        if group in parameters.keys():
            if parameters[group] == True:
                selected_features.extend(variable_sets[group])

    return sorted(selected_features)


    
def update_config_tunning(config, parameters):
    # Special Kind of tunning
    if config['opt_groups']:
        selected_features = get_features(parameters, config['variable_groups'])
        config['dataset_generator_params']['all_features'] = selected_features
        
    # General parameters
    for key, path in config['params_to_opt'].items():
        __setInDict(config, path.split('/'), parameters[key])
    
    return config