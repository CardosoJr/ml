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
from .dnn import DNNBuilder


class TorchStackingDNN(nn.Module):
    """
    Class representing Deep Stacking Network in pytorch 
 
    paper: https://ieeexplore.ieee.org/abstract/document/6389679/
 
    """
    
    def __init__(self, model_params):
        super().__init__()

        
        self.blocks = []
        initial_size = model_params['initial_size']
        self.blocks.append(self.build_block(model_params = model_params,
                                        size = model_params['layer_size'],
                                        input_size = initial_size))
        
    
        for index in np.arange(1, model_params['num_blocks']):
            initial_size = initial_size + 1 
            self.blocks.append(self.build_block(model_params = model_params,
                                        size = model_params['layer_size'],
                                        input_size = initial_size))
            
            
        # Initializations 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def build_block(self, model_params, size, input_size):
        layers = []
        layers.append(nn.Linear(input_size, size))
        layers.append(self.get_activation(model_params['activation']))
        
        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))
            
        layers.append(nn.Linear(size, 1))
        
        return nn.Sequential(*layers)
    
    def get_activation(self, activation):
        if activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        elif activation.lower() == 'selu':
            return nn.SELU()
        else:
            return nn.ReLU()
            
    def forward(self, x):
        current_input = x
        for block in self.blocks:
            out = block(current_input)
            current_input = torch.cat((current_input, out), dim = 0)
        return out

class SDNBuilder(DNNBuilder):
    """
    Builds Stacked DNNs model
    """

    def __init__(self, 
                 model_params = {},
                 ds_params = {}):
        
        super().__init__(model_params, ds_params)
        

    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = SDNBuilder(**config)
        dnn.build_model()
        dnn.model.load_state_dict(torch.load(path + "/model.pt"))
        return dnn
            
    def build_model(self):
        self.model = TorchStackingDNN(self.model_params).to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss() # Sigmoid and BCE loss  ## nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience = self.model_params['early_stopping_it'])