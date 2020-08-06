import pandas as pd
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc
from pathlib import Path
import os

from generic_model.ml_pipe.Tasks import aux_tasks
import generic_model.tool_config.config_methods as config_methods

class TaskCreateTrainEvalDS(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Creates Train e Eval dataframes 
    input:
        create_ds: CreateDS object
        ds_params: params for creating model dfs
        
    output:
        'train_df' : train datafrmae
        'eval_df' : eval dataframe
        train_eval_keys : train and eval samples keys 
    '''
    
    taks_te_params = luigi.Parameter()

    persist = ['train_df', 'eval_df', 'train_eval_keys'] 
    
    def run(self):        
        train, eval, keys = config_methods.train_eval_creator[self.taks_te_params['method']](self.taks_te_params['create_ds'])\
                                           .create_train_eval_bq(**self.taks_te_params['ds_params'])
        output = {}
        output['train_df'] = train
        output['eval_df'] = eval
        output['train_eval_keys'] = keys
        self.save(output)

class TaskCreatePredictScoringDS(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Creates Predict and Scoring DataFrames 
        
    input:
        create_ds: CreateDS object
        ds_params: params for creating model dfs
        
    output:
        'predict_df' : train datafrmae
        'scoring_df' : eval dataframe
    
    '''
    task_ps_params = luigi.Parameter()
    persist = ['predict_df', 'scoring_df'] 
    
    def run(self):
        predict, scoring = config_methods.predict_score_creator[self.task_ps_params['method']](self.task_ps_params['create_ds'])\
                                          .create_predict_scoring_bq(**self.task_ps_params['ds_params'])
        output = {}
        output['predict_df'] = predict
        output['scoring_df'] = scoring
        self.save(output) 
        
        
@d6tflow.inherits(TaskCreateTrainEvalDS)
@d6tflow.inherits(TaskCreatePredictScoringDS)
class TaskPrepareDS(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        All remaining feature engineering needed to be done before for the model. 
        If possible, most of all feature engineering should have been done in source data already (much faster if done in data warehouse like BQ)
        This Task receives a list of engineering objects and treats them sequentially (maybe in the future we should create multiple tasks to do this)
    input:
        'engineer' : engineer object 
        'engineer_params' : parameters for the engineer object
        
    output:
        train_df : processed train dataframe
        eval_df : processed eval dataframe
        predict_df :processed predict dataframe
        scoring_df : processed scoring dataframe
    '''
    
    task_engineer_params = luigi.Parameter()
    persist = ['train_df','eval_df', 'predict_df', 'scoring_df'] 
    
    def requires(self):
        return {'train_eval': self.clone(TaskCreateTrainEvalDS), 'predict_score': self.clone(TaskCreatePredictScoringDS)}
    
    def run(self):
        output = {
            'train_df'   : self.input()['train_eval']['train_df'].load(),
            'eval_df'    : self.input()['train_eval']['eval_df'].load(),
            'predict_df' : self.input()['predict_score']['predict_df'].load(),
            'scoring_df' : self.input()['predict_score']['scoring_df'].load()
        }

        for method in self.task_engineer_params['method']:
            output = config_methods.engineer_creator[method](self.task_engineer_params['engineer'][method]).prepare(**output)
            
        self.save(output)
        
        
@d6tflow.inherits(TaskPrepareDS) 
@d6tflow.clone_parent        
class TaskTrainModel(aux_tasks.TaskModel):
    '''
    description:
        training ML model
    input:
        'model' : model object to be executed
        'build_params'  : model params dict
    
    output:
        model
    '''
    task_model_params = luigi.Parameter()
    def run(self):
        model = config_methods.model_creator[self.task_model_params['method']](self.task_model_params['model'])
        model.build(
            datasets = {'train_df'  : self.input()['train_df'].load(), 
                       'eval_df'    : self.input()['eval_df'].load(),
                       'predict_df' : self.input()['predict_df'].load(),
                       'scoring_df' : self.input()['scoring_df'].load()},
            params = self.task_model_params['build_params'])
        self.save(model)
            
@d6tflow.inherits(TaskTrainModel) 
@d6tflow.inherits(TaskPrepareDS) 
class TaskPredict(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Predicts probabilities to predict and scoring dataframes
    input:
        'predict_params'
    output:
    
    '''
    task_predict_params = luigi.Parameter()
    def requires(self):
        return {'model': self.clone(TaskTrainModel), 'engineer':self.clone(TaskPrepareDS)}
            
    def run(self):
        model = self.input()['model'].load()
        predict_df = self.input()['engineer']['predict_df'].load()
        scoring_df = self.input()['engineer']['scoring_df'].load()
        
        scores = model.predict(datasets  = {'predict_df' : predict_df, 'scoring_df' : scoring_df},
                     params = self.task_predict_params['predict_params'])
        
        self.save(scores)
            
            
@d6tflow.inherits(TaskPredict) 
@d6tflow.clone_parent  
class TaskModelMetrics(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        calculate model metrics for particular run 
    input:
        metrics : dict of metric functions 
    output:
        dataframe with metrics
    '''
    task_metrics_params = luigi.Parameter()
    persist = ['full_metrics']
    
    def run(self):
        scores = self.input().load()
        
        print(self.task_metrics_params['score_params'])
        
        output = {}
            
        output['full_metrics'] = pd.DataFrame([])
        
        for method in self.task_metrics_params['method']:
            metric_obj = config_methods.metrics_creator[method](self.task_metrics_params['metrics'][method])
            output['full_metrics'] = output['full_metrics'].append(metric_obj.calculate(scores, self.task_metrics_params['score_params']), ignore_index = True)
        
        self.save(output)

##################################################################################################
########################################## ELASTICITY ############################################
##################################################################################################
    
@d6tflow.inherits(TaskPrepareDS) 
@d6tflow.clone_parent  
class TaskPrepareElasticityDataset(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Prepares the dataset to calculate the elasticity values
    input:
    output:
    '''
    task_el_ds_params = luigi.Parameter()
    persist = ['elasticity_df', 'elasticity_scoring_df', 'elasticity_real_df']
    
    def run(self):
        predict_df = self.input()['predict_df'].load()
        scoring_df = self.input()['scoring_df'].load()
        
        mask = (scoring_df[self.task_el_ds_params['date_col']] >= self.task_el_ds_params['elasticity_begin']) & (scoring_df[self.task_el_ds_params['date_col']] < self.task_el_ds_params['elasticity_end'])
            
        predict_df = predict_df[mask]
        scoring_df = scoring_df[mask]
        
        pred_cols = list(predict_df.columns)

        not_in = [x for x in list(scoring_df.columns) if x not in pred_cols]
        
        elasticity_df = pd.concat([predict_df, scoring_df[not_in]], axis = 1)
        
        new_elast_df = config_methods.elasticity.transform_df(elasticity_df, 
                                                              self.task_el_ds_params['n_min'],
                                                              self.task_el_ds_params['n_max'], 
                                                              self.task_el_ds_params['qtd_pass'])
        
        # saving elasticity df
        output = {}
        output['elasticity_df'] = new_elast_df.drop(columns = not_in + ["FATOR"])
        output['elasticity_scoring_df'] = new_elast_df[list(scoring_df.columns) + ["FATOR"]]
        output['elasticity_real_df'] = pd.DataFrame([])
        
        self.save(output)
    
@d6tflow.inherits(TaskTrainModel) 
@d6tflow.inherits(TaskPrepareElasticityDataset) 
class TaskElasticity(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Predicts probabilities to predict and scoring dataframes
    input:
        'predict_params'
    output:
    
    '''
    task_elasticity_params = luigi.Parameter()
    def requires(self):
        return {'model': self.clone(TaskTrainModel), 'elasticity' : self.clone(TaskPrepareElasticityDataset)}
            
    def run(self):
        model = self.input()['model'].load()
        predict_df = self.input()['elasticity']['elasticity_df'].load()
        scoring_df = self.input()['elasticity']['elasticity_scoring_df'].load()
        
        scores = model.predict(datasets  = {'predict_df' : predict_df, 'scoring_df' : scoring_df},
                     params = self.task_elasticity_params['predict_params'])

        
        # Fixing columns for Elasticity Report 
        scores['MODEL_NAME'] = self.task_elasticity_params['model_name']
        self.save(scores)          
        
@d6tflow.inherits(TaskElasticity) 
@d6tflow.inherits(TaskPrepareDS) 
class TaskElasticityReport(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Predicts probabilities to predict and scoring dataframes
    input:
        'predict_params'
    output:
    
    '''
    task_elasticity_report_params = luigi.Parameter()
    persist = ['real_df', 'model_df']
    
    def requires(self):
        return {'original_df': self.clone(TaskPrepareDS), 'elasticity_df' : self.clone(TaskElasticity)}
            
    def run(self):
        scoring = self.input()['original_df']['scoring_df'].load()
        el_df = self.input()['elasticity_df'].load()
        
        output = {}
        print(scoring.columns)
        print(el_df.columns)
        
        
        #Atualizar para a função agregate do Report !!
        output['real_df'] = config_methods.elasticity.compile_report(df = scoring, 
                                                             n_min = self.task_elasticity_report_params['n_min'], 
                                                             n_max = self.task_elasticity_report_params['n_max'], 
                                                             step = self.task_elasticity_report_params['real_qtd_pass'], 
                                                             output = self.task_elasticity_report_params['target'],
                                                             model_name = "REAL")
        
        output['model_df'] = config_methods.elasticity.compile_report(df = el_df, 
                                                             n_min = self.task_elasticity_report_params['n_min'], 
                                                             n_max = self.task_elasticity_report_params['n_max'], 
                                                             step = self.task_elasticity_report_params['qtd_pass'], 
                                                             output = self.task_elasticity_report_params['output'], 
                                                             model_name = self.task_elasticity_report_params['model_name'])
        
        self.save(output)     
            
##################################################################################################
###################################### BACKTESTING TASKS #########################################
##################################################################################################
    
class BacktestReport(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        consolidates results from all executions in a backtest
    input:
    output:
    '''
    task_bkt_params = luigi.Parameter()
    persist = ['full_metrics']
    
    def requires(self):
        return [TaskModelMetrics(**el) for el in self.task_bkt_params]
        
    def run(self):
        full_metrics = pd.DataFrame([])
        
        for index, inpt in enumerate(self.input()):
            key = self.task_bkt_params[index]['taks_te_params']['key']
            
            inpt_full_metrics = inpt['full_metrics'].load()
            
            inpt_full_metrics['period'] = [key] * len(inpt_full_metrics) # key
            inpt_full_metrics['model_name'] = [self.task_bkt_params[index]['task_metrics_params']['model_name']] * len(inpt_full_metrics)
            
            full_metrics = full_metrics.append(inpt_full_metrics, ignore_index = True)
            
        output = {'full_metrics' : full_metrics}
        self.save(output)