import graphviz
import importlib
import numpy as np

from algorithms.wrapper import Wrapper
from feature.manipulator import Manipulator
from utils import flip_dict, load_inv_column_map

from django.utils.text import slugify
from collections import OrderedDict
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


class Pipeline:
    """
    This class is a bit like sklearn's Pipeline, used when you need to shoehorn a bunch of algorithms/
    transformers (anything with fit/transform) methods in before a final algorithm. Helpful for when
    it's necessary to fork data processing off the main pipeline.

    If using multiple Pipeline objects (like in oos_ensemble), prefix each with a number plus '__'
    otherwise it will cause a problem.

    Make sure final algorithm has a predict method in addition to fit/transform.
    """

    def __init__(self, algorithms):
        pipeline_algorithms = OrderedDict()
        for algo_dict in algorithms:
            algo_name = algo_dict.keys()[0]
            fully_qualified_algo = max(algo_name.split('__'), key=len)
            module_name = '.'.join(fully_qualified_algo.split('.')[:-1])
            algo_method = fully_qualified_algo.split('.')[-1]
            algo_module = importlib.import_module(module_name)
            algo_class = getattr(algo_module, algo_method)
            kwargs = algo_dict[fully_qualified_algo]['keyword_arg_settings']
            algo_instance = algo_class(**kwargs)
            pipeline_algorithms[fully_qualified_algo] = algo_instance
        self.pipeline_algorithms = pipeline_algorithms

    def fit(self, X, y):
        pipeline_algorithms = self.pipeline_algorithms
        len_pipeline = len(pipeline_algorithms)
        X_t, y_t = X, y
        for i in range(len_pipeline):
            name, algo = pipeline_algorithms.items()[i]
            algo.fit(X_t, y_t)
            if i < (len_pipeline - 1):
                X_t  = algo.transform(X_t)
        self.pipeline_algorithms = pipeline_algorithms

    def transform(self, X, y):
        pipeline_algorithms = self.pipeline_algorithms
        X_t, y_t = X, y
        for name, algo in pipeline_algorithms.items():
            X_t, y_t = algo.transform(X_t, y_t)
        return X_t, y_t

    def predict(self,X):
        pipeline_algorithms = self.pipeline_algorithms
        name, last_algo = pipeline_algorithms.items()[-1]
        return last_algo.predict(X)

class PassThrough:
    """Used for meta-modeling, when prediction already exists as a feature in the X matrix"""
    def __init__(self):
        pass

    def fit(self,X,y):
        #TODO: Add some sort of helpful error here for when metamodeler is specified without kaggle_stack
        assert X.shape[1] == 1

    def predict(self,X):
        assert X.shape[1] == 1
        return np.array(X[0])