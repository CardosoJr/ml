import tensorflow as tf
import dill
from generic_model.ml_pipe.Models import dnn, xgb, sdnn
from generic_model.ml_pipe.Models import pytorch

def save_model(obj, path):
    obj.save_model(str(path))

def load_model(path):
    
    with open(str(path) + '/model_class.pkl', 'rb') as f:
        model_class = dill.load(f)
        
    if model_class['model'] == 'dnn':
        return dnn.DNN.load_model(str(path))
    elif model_class['model'] == 'sdnn':
        return sdnn.SDNN.load_model(str(path))
    
    elif model_class['model'] == 'xgboost':
        return xgb.XGB.load_model(str(path))
    
    elif model_class['model'] == 'dnn_pt':
        return pytorch.dnn.DNNBuilder.load_model(str(path))
    elif model_class['model'] == 'sdnn_pt':
        return pytorch.sdnn.SDNBuilder.load_model(str(path))
