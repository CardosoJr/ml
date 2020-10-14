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
from generic_model.ml_pipe.Models.pytorch.utils import  EarlyStopping, binary_acc
from .utils import MtlDFDataset
from generic_model.ml_pipe import utils
from generic_model.ml_pipe.Models.pytorch.dnn import DNNBuilder


class TorchMDNN(nn.Module):
    '''
    Represents torch multi-task dnn
    '''
    
    def __init__(self, model_params):
        super(TorchDNN, self).__init__()
        self.block1, output_size = self.build_block(model_params = model_params,
                                        size = model_params['fb_initial_size'],
                                        num_layers = model_params['fb_layers'], 
                                        rate  = model_params['fb_rate'], 
                                        min_size =  model_params['fb_min_size'], 
                                        input_size = model_params['initial_size'])
        
        if model_params['sb_layers'] > 0:
            self.block2, output_size =  self.build_block(model_params = model_params,
                                            size = model_params['sb_initial_size'],
                                            num_layers = model_params['sb_layers'], 
                                            rate  = model_params['sb_rate'], 
                                            min_size =  model_params['sb_min_size'],
                                            input_size = output_size)
        
        else:
            self.block2 = None
        
        self.final_layers = []
        for i in np.arange(model_params['num_objectives']):
            self.final_layers.append(nn.Linear(output_size, 1))
        
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
        
    
    def build_block(self, model_params, size, num_layers, rate, min_size, input_size):
        layers = []
        layers.append(nn.Linear(input_size, size))
        layers.append(self.get_activation(model_params['activation']))
        previous_size = int(size)
        
        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))
            
        for nl in np.arange(num_layers - 1):
            size = int(size * rate)
            if size < min_size: 
                size = min_size
            
            layers.append(nn.Linear(previous_size, size))
            previous_size = size
            
            layers.append(self.get_activation(model_params['activation']))
            if model_params['dropout'] > 0:
                layers.append(nn.Dropout(model_params['dropout']))
            
        return nn.Sequential(*layers), size
    
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
        out = self.block1(x)
        if self.block2 is not None:
            out = self.block2(out)
            
        logit = []
        for layer in self.final_layers:
            logit.append(layer(out))
        return logit # nn.functional.sigmoid(logit)

class NaiveMDNNBuilder(DNNBuilder):
    """
    Builds MTL DNNs model
    """

    def __init__(self, 
                 model_params = {},
                 ds_params = {}):
        
        super().__init__(model_params, ds_params)
        

    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = NaiveMDNNNBuilder(**config)
        dnn.build_model()
        dnn.model.load_state_dict(torch.load(path + "/model.pt"))
        return dnn
    
    def get_model_name(self):
        return 'mtl_dnn_pt'
    
    def get_loss_fn(self, func):
        if func == 'bce':
            return nn.BCEWithLogitsLoss()
        
    def build_model(self):
        self.model = TorchMDNN(self.model_params)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = [self.get_loss_fn(func) for func in self.model_params['losses']]
        
        if self.model_params['early_stopping_it'] < 0:
            patience = 1e9
        else:
            patience = self.model_params['early_stopping_it']
            
        self.early_stopping = EarlyStopping(patience = patience)
        
    
    def build_dataset(self, df, label, batch_size = 32, use_gpu = False, training = True):
        loader = torch.utils.data.DataLoader(MtlDFDataset(df, label), 
                                             batch_size = batch_size,
                                             shuffle = training, 
                                             pin_memory = use_gpu,
                                             num_workers = 4 if training else 1)
        return loader   
    
        
    def train(self, train_loader, test_loader, epoch):
        running_loss = 0
        epoch_loss = 0
        metrics = None
        epoch_metrics = None
        self.model.train()
        for batch_idx, full_data in enumerate(train_loader):
            data = full_data[0].to(self.device)
            targets = [x.to(self.device) for x in full_data[1:]]

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = sum([self.loss_fn[i](o, target.float().unsqueeze(1)) * self.model_params['loss_weights'] for i, o in enumerate(output)])
            loss.backward()
            self.optimizer.step()            
            running_loss += loss.item()
            epoch_loss += loss.item()
            
            if self.model_params['visualize']:
                metrics = self.calculate_metrics(torch.sigmoid(output[0]).squeeze(1), targets[0].float(), metrics)
                epoch_metrics = copy.deepcopy(metrics)
            
            if batch_idx > 0 and self.model_params['log_interval'] > 0 and batch_idx % self.model_params['log_interval'] == 0:
                if self.model_params['visualize']:
                    val_loss, val_metrics = self.test(test_loader)
                    metrics = self.normalize_metrics(metrics)
                    
                    self.tb_log_metrics({"loss" : running_loss / metrics['counter'], **metrics},
                                          {"loss" : val_loss, **val_metrics},
                                          epoch * len(train_loader) + batch_idx + 1)
                
                    metrics = None # reseting metrics calculations
                    running_loss = 0 # reseting loss
                    self.model.train() ## get back to training mode
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        return epoch_loss / len(train_loader), self.normalize_metrics(epoch_metrics)
             
    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        metrics = None
        with torch.no_grad():
            for full_data in test_loader:
                data = full_data[0].to(self.device)
                targets = [x.to(self.device) for x in full_data[1:]]
                output = self.model(data)
                test_loss += sum([self.loss_fn[i](o, target.float().unsqueeze(1)) * self.model_params['loss_weights'] for i, o in enumerate(output)])
                
                if self.model_params['visualize']:
                    metrics = self.calculate_metrics(torch.sigmoid(output[0]).squeeze(1), targets[0].float(), metrics)

        test_loss /= len(test_loader)
        print('\nTest set: Avg loss: {:.4f}\n'.format(test_loss))
        
        metrics = self.normalize_metrics(metrics)
        
        return test_loss, metrics
        
    def get_output(self, predict_loader):
        y_pred_list = np.array([])
        self.model.eval()
        metrics = None
        with torch.no_grad():
            for data in predict_loader:
                data = data.to(self.device)
                output = torch.sigmoid(self.model(data)[0])
                y_pred_list = np.append(y_pred_list, output.cpu().detach().numpy().squeeze())

        return y_pred_list
        
    def predict(self, datasets, params):
        '''
        description: 
            uses model to make predictions
        input: 
            datasets {'predict_df': dataframe, 'scoring_df' : dataframe}
        output:
        '''
        
        X_predict = datasets['predict_df']
        scoring = datasets['scoring_df']
        y_predict = X_predict[self.ds_params['target']]
        X_predict = X_predict.drop(columns = self.ds_params['target'])
        
        predict_dataset = self.build_dataset(X_predict, 
                                               None, 
                                               batch_size = self.model_params['batch_size'], 
                                               training = False,
                                               use_gpu = self.model_params['gpu'])
        
        
        y_pred = self.get_output(predict_dataset)
        
        scoring['PREDICTED'] = y_pred

        scoring = scoring.rename(columns = {
            self.ds_params['date_col'] : 'DATE',
            self.ds_params['target'] : 'TARGET',
        })
        
        return scoring