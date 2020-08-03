import d6tflow
import pickle
import dill
import cloudpickle

from generic_model.ml_pipe.Models import helpers

class PickleTarget2(d6tflow.targets.DataTarget):
    """
    Saves to pickle, loads to python obj

    """
    def load(self, cached=False, **kwargs):
        """
        Load from pickle to obj

        Args:
            cached (bool): keep data cached in memory
            **kwargs: arguments to pass to pickle.load

        Returns: dict

        """
        def load_aux_func(x):
            with open(x, 'rb') as f:
                 data = dill.load(f)
            return data
        
        return super().load(lambda x: load_aux_func(x), cached, **kwargs)
    
    def save(self, obj, **kwargs):
        """
        Save obj to pickle

        Args:
            obj (obj): python object
            kwargs : additional arguments to pass to pickle.dump

        Returns: filename

        """
        (self.path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.path, "wb") as f:
            dill.dump(obj, f, **kwargs)
        return self.path
    
class TaskPickle2(d6tflow.tasks.TaskData):
    """
    Task which saves to pickle
    """
    target_class = PickleTarget2
    target_ext = 'pkl'
    
    
class ModelTarget(d6tflow.targets.DataTarget):
    """
    For keras models

    """    
    def load(self, cached=False, **kwargs):
        """
        Load from pickle to obj

        Args:
            cached (bool): keep data cached in memory
            **kwargs: arguments to pass to pickle.load

        Returns: dict

        """
        return super().load(helpers.load_model, cached, **kwargs)
        
    def save(self, obj, **kwargs):
        """
        Save obj to pickle

        Args:
            obj (obj): python object
            kwargs : additional arguments to pass to pickle.dump

        Returns: filename

        """
        (self.path).parent.mkdir(parents=True, exist_ok=True)
        helpers.save_model(obj, self.path)
        return self.path
    
class TaskModel(d6tflow.tasks.TaskData):
    """
    Task which saves to pickle
    """
    target_class = ModelTarget
    target_ext = '' 