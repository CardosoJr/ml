import sys
import d6tflow
from multiprocessing import Process
import os
import logging
import logging.config
import yaml


if __name__ == '__main__':
    if sys.argv[1] == 'generic_ds':
        from generic_model.methods.dataset import ds_generator 
        ds_generator.run(sys.argv[2])
            
    elif sys.argv[1] == 'generic_model':
        if sys.argv[2] == "dev":
            if sys.argv[3] == 'run_backtest':
                from generic_model.methods.backtest import backtest_run as bt
                print('Running Backtest')
                bt.run(sys.argv[4])        

            elif sys.argv[3] == 'run_hyperopt_search':
                from generic_model.methods.hyperopt_search import gridsearch_run as hs
                print('Running Hyperopt')
                hs.run(sys.argv[4])
                    
    else:
        print('No found methods')        
