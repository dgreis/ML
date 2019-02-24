import numpy as np

from algorithms.wrapper import Wrapper


class MetaModeler(Wrapper):

    def __init__(self, model_config, project_settings, mode='algorithm'):
        base_algo_class = PassThrough
        super(MetaModeler, self).__init__(base_algo_class, model_config, project_settings, mode)


class PassThrough:
    """Used for meta-modeling, when prediction already exists as a feature in the X matrix"""
    def __init__(self):
        pass

    def fit(self,X,y):
        assert X.shape[1] == 1

    def predict(self,X):
        assert X.shape[1] == 1
        return np.array(X[0])