import multiprocessing
import tensorflow as tf
from datetime import datetime
import os
import yaml

class DeviceManager:
    """
    Device Manager:  class to manage access to GPU
    """
    
    def __init__(self):
        with open('gpu_config.yaml', 'rb') as f:
            config = yaml.safe_load(f)
        self.__set_environ(config['ENV_VARS'])
        self.gpu_devices = config['GPUS']
        self.max_num_process = config['MAX_PROCESSES_PER_GPU']

        self.lock = multiprocessing.Lock()

        self.usage = {}
        for gpu in self.gpu_devices:
            self.usage[gpu] = 0

    def __set_environ(self, vars):
        for k, v in vars.items():
            os.environ[k] = v            
    
    def get_available_device(self):
        device = None
        
        self.lock.acquire()
        device = None
        try:
            available = [x for x,y in self.usage.items() if y < self.max_num_process]
            if len(available) > 0:
                device = [available[0]]
                self.usage[available[0]] = self.usage[available[0]] + 1
            
        except Exception as ex:
            print('ERROR')
            print(type(ex))   
            print(ex.args)     
            print(ex)       
        finally:
            self.lock.release()
        
        if device == None:
            device = []
        
        return device 
    
    def release(self, devices):
        self.lock.acquire()
        try:
            for device in devices:
                if device in self.usage.keys():
                    self.usage[device] -= 1
                    
        except Exception as ex:
            print("ERROR")
            print(type(ex))   
            print(ex.args)     
            print(ex)       
        finally:
            self.lock.release()
         