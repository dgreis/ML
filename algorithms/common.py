import graphviz

import numpy as np

from algorithms.wrapper import Wrapper
from utils import flip_dict, load_inv_column_map

from django.utils.text import slugify
from sklearn import tree


class DecisionTree(Wrapper):

    def __init__(self, wrapper_id, base_algo_class, model_config, project_settings):
        super(DecisionTree, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

    def gen_output(self):
        clf = self.base_algorithm
        prior_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        inv_col_map = load_inv_column_map(prior_feature_names_filepath)
        col_map = flip_dict(inv_col_map)
        num_cols = len(col_map)
        column_names = [col_map[i] for i in range(num_cols)]
        dot_data = tree.export_graphviz(clf, out_file=None,feature_names=column_names)
        graph = graphviz.Source(dot_data,format='png')
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        graph.render(filename=artifact_dir +'/' + slugify(model_name) + '-tree' )

class MetaModeler(Wrapper):

    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = PassThrough
        super(MetaModeler, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)


class PassThrough:
    """Used for meta-modeling, when prediction already exists as a feature in the X matrix"""
    def __init__(self):
        pass

    def fit(self,X,y):
        assert X.shape[1] == 1

    def predict(self,X):
        assert X.shape[1] == 1
        return np.array(X[0])