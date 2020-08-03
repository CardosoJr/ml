import pandas as pd
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import d6tflow
import luigi
import gc

from modelos.generic_model.ml_pipe.Tasks import aux_tasks
import modelos.generic_model.generic_tasks as tasks
import modelos.generic_model.tool_config.config_methods as config_methods

##################################################################################################
########################################## ELASTICITY ############################################
##################################################################################################
    
@d6tflow.inherits(tasks.TaskPrepareDS) 
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
    
@d6tflow.inherits(tasks.TaskTrainModel) 
@d6tflow.inherits(tasks.TaskPrepareElasticityDataset) 
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
@d6tflow.inherits(tasks.TaskPrepareDS) 
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