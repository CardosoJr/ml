from torch.utils.data import Dataset
from torch import nn, optim
import torch

class MtlDFDataset(Dataset):
    def __init__(self, df, target_fields):
        self.has_target  = False
        if target_field is not None:
            self.has_target = True
            self.target_fields = target_fields
            self.target = df[target_fields] # astype("float32")
            self.data = df.loc[:, [a not in target_fields for a in df.columns]].astype("float32").values
            
        else:
            self.data = df.astype("float32").values
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.has_target:
            data = [self.data[idx]]               
            for tgt in self.target_fields:
                data.append(self.target[tgt].values[idx])
            return data
        else:
            return self.data[idx]