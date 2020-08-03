import pandas as pd
import numpy as np
import math
import os
import dill
from modelos.generic_model.tool_config import config_methods
from datetime import datetime
import os
from torch import nn, optim
import torch
from .utils import DFDataset, EarlyStopping, binary_acc
from generic_model.ml_pipe import utils
from .dnn import DNN

class TorchResDNN(nn.Module):
    """
    Residual DNN in pytorch 
    
    this represents Deep Neural Network to run with ml-pipe in d6tflow 
    
    TODO: 
    * Custom loss / metrics
    * Checkpoint
    * Tensorboard
    * Class weight
    * Learning Rate decay
    
    Known Bugs: 
    Got error when pickling this class due to weakref. Not sure which object, maybe the strategy one. 
    Using Dill (instead of pickle) to deal with this, but gotta investigate more, if this will impact further
    """
    
    def __init__(self, model_params):
        super(TorchDNN, self).__init__()
        self.block1, output_size = self.__build_block(model_params = model_params,
                                        size = model_params['fb_initial_size'],
                                        num_layers = model_params['fb_layers'], 
                                        rate  = model_params['fb_rate'], 
                                        min_size =  model_params['fb_min_size'], 
                                        input_size = model_params['initial_size'])
        
        if model_params['sb_layers'] > 0:
            self.block2, output_size =  self.__build_block(model_params = model_params,
                                            size = model_params['sb_initial_size'],
                                            num_layers = model_params['sb_layers'], 
                                            rate  = model_params['sb_rate'], 
                                            min_size =  model_params['sb_min_size'],
                                            input_size = output_size)
        
        else:
            self.block2 = None
        
        self.final_layer = nn.Linear(output_size, 1)
    
    def __build_block(self, model_params, size, num_layers, rate, min_size, input_size):
        layers = []
        layers.append(nn.Linear(input_size, size))
        layers.append(self.__get_activation(model_params['activation']))
        previous_size = int(size)
        
        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))
            
        for nl in np.arange(num_layers - 1):
            size = int(size * rate)
            if size < min_size: 
                size = min_size
            
            layers.append(nn.Linear(previous_size, size))
            previous_size = size
            
            layers.append(self.__get_activation(model_params['activation']))
            if model_params['dropout'] > 0:
                layers.append(nn.Dropout(model_params['dropout']))
            
        return nn.Sequential(*layers), size
    
    def __get_activation(self, activation):
        if activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        elif activation.lower() == 'selu':
            return nn.SELU()
        else:
            return nn.ReLU()
            
    def forward(self, x):
        out = self.block1(x)
        if self.block2 is not None:
            out = self.block2(out)
        logit = self.final_layer(out)
        return logit # nn.functional.sigmoid(logit)
        

class ModelBuilder(DNN):
    """
    Builds Res DNNs modelaa
    """

    def __init__(self, 
                 model_params = {},
                 ds_params = {}):
        
        super(B,self).__init__(model_params, ds_params)
        

    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = ModelBuilder(**config)
        dnn.__build()
        dnn.model.load_state_dict(torch.load(path + "/model.pt"))
        return dnn
            
    def __build(self):
        self.model = TorchResDNN(self.model_params).to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss() # Sigmoid and BCE loss  ## nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience = self.model_params['early_stopping_it'])