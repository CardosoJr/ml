import pandas as pd
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import d6tflow
import luigi
import gc
from datetime import datetime
from pathlib import Path
import os


from modelos.generic_model.ml_pipe.Tasks import aux_tasks
from modelos.generic_model.ml_pipe.Optimization import optimizer
from lib import Elasticity_Transform
import modelos.generic_model.tool_config.config_methods as config_methods
from generic_tasks import *


@d6tflow.inherits(TaskElasticity) 
@d6tflow.inherits(TaskModelMetrics) 
@d6tflow.inherits(TaskPredict)
class TaskOptimize(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        Optimizing margin
    input:
        'model' : model object to be executed
        'build_params'  : model params dict
    
    output:
        model
    '''
    task_opt_params = luigi.Parameter()
    
    def requires(self):
        return {'metric': self.clone(TaskModelMetrics), 'elasticity' : self.clone(TaskElasticity), 'predict' : self.clone(TaskPredict)}
    
    def run(self):
        scores = self.input()['elasticity'].load()
        original_scores = self.input()['predict'].load()
        
        scores.rename(columns = {
            self.task_opt_params['output'] : 'prob_renov',
            self.task_opt_params['key'] : 'u_id',
            'FATOR' : 'ajuste'},
                      inplace = True)
        
        # ADD COMISS TO CALCULATE MARGIN 
        scores['obj'] = (scores['obj'] * scores['predicted']
        scores = scores[(scores[self.task_opt_params['date_col']] >= self.task_opt_params['prediction_begin']) & (scores[self.task_opt_params['date_col']] < self.task_opt_params['prediction_end'])]

        original_scores = original_scores[(original_scores[self.task_opt_params['date_col']] >= self.task_opt_params['prediction_begin']) & (original_scores[self.task_opt_params['date_col']] < self.task_opt_params['prediction_end']) ]
        
        original_scores['obj'] = original_scores['obj'] * original_scores[self.task_opt_params['target']]

        original_conversion = round(original_scores[self.task_opt_params['target']].mean(), 3)
        
        scores_to_opt = scores[['u_id', 'prob_renov', 'ajuste', 'obj']].copy(deep = True)
          
        opt  = optimizer.Optimizer(lock = config_methods.optimization_lock, optimizer_name = 'gurobi')
        
        df_score_out, obj_value, global_conversion, _, _ = opt.optimize(df_curvas_elasticidade = scores_to_opt,
                                                                        min_global_conversion = original_conversion + 1e-05)
        
        data = {'model' : self.task_opt_params['train_tag'], 'prediction' : self.task_opt_params['prediction_tag'], 'opt_margin' : obj_value, 'opt_conversion' : global_conversion, 'real_margin' : original_scores['obj'].sum(), 'real_conversion' : original_conversion}
        
        metric = self.input()['metric']['full_metrics'].load()
        metric = metric[metric['period'] == 'erro_oos']
        for err in self.task_opt_params['metrics']:
            err_df = metric[metric['metric'] == err]
            data[err + "_mean"] = err_df['value'].mean()
            data[err + "_std"] = err_df['value'].std()
        
        result = pd.DataFrame([data])
        self.save(result)

class TaskOptSummary(d6tflow.tasks.TaskPqPandas):
    '''
    description:
        consolidates results from all optimizations
    input:
    output:
    '''
    task_opt_summary_params = luigi.Parameter()
    def requires(self):
        return [TaskOptimize(**el) for el in self.task_opt_summary_params]
        
    def run(self):
        full_metrics = pd.DataFrame([])
        for index, inpt in enumerate(self.input()):
            inpt_full_metrics = inpt.load()
            full_metrics = full_metrics.append(inpt_full_metrics, ignore_index = True)
            
        self.save(full_metrics)