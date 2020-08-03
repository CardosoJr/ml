import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import yaml

from generic_model.ml_pipe.Utils import BigqueryUtils
from generic_model.ml_pipe.FeatureEngineering.utils import febq

def handle_config(config_path):
    with open(config_path, 'r') as stream:
        config_data = yaml.safe_load(stream)
    return config_data
    
def run(config_path):
    config = handle_config(config_path)
    input_table = "`" + config['table_name'] + "`"
    
    for operation_dict in config['operations']:
        for operation, param in operation_dict.items():
            print('Doing operation', operation)

            param = {**param, 'table_name' : input_table}
            input_table = "`" + config['destination_table'] + "`" 

            sql = febq.methods_dict[operation](**param)
            BigqueryUtils.create_table_from_query(query           = sql,
                                                  project_name    = config['destination_table'].split(".")[0], 
                                                  dataset_name    = config['destination_table'].split(".")[1], 
                                                  table_name      = config['destination_table'].split(".")[2] ,
                                                  expiration_days = None, 
                                                  append          = False)
    
