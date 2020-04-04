
from utils import flip_dict, load_inv_column_map
from .wrapper import Wrapper
from .common import DecisionTree
from sklearn import tree
from sklearn import ensemble
from sklearn import multiclass
from sklearn import base
from algorithms.algoutils import get_algo_class

#TODO: Do I need this below? Or is it redundant?
class DecisionTreeClassifier(DecisionTree):

    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = tree.DecisionTreeClassifier
        super(DecisionTreeClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

class VotingClassifier(Wrapper):
    """Custom implementation of sklearn.ensemble.VotingClassifier because component classifiers need
    to be initialized first.

    """
    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = ensemble.VotingClassifier
        estimators_raw = model_config['keyword_arg_settings']['estimators']
        if type(estimators_raw) == dict:
            estimators_final = list()
            for k in estimators_raw:
                est_algo_class = get_algo_class(estimators_raw[k]['base_algorithm'])
                est_algo_kwargs = estimators_raw[k]['keyword_arg_settings']
                est_algo_instance = est_algo_class(**est_algo_kwargs)
                estimators_final.append((k,est_algo_instance))
        elif type(estimators_raw) == list:
            estimators_final = estimators_raw
        else:
            raise Exception
        model_config['keyword_arg_settings']['estimators'] = estimators_final
        super(VotingClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

class OutputCodeClassifier(Wrapper):
    """Custom implementation of sklearn.multiclass.OutputCodeClassifier because component classifiers need
    to be initialized first.

    """
    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = multiclass.OutputCodeClassifier
        estimator_raw = model_config['keyword_arg_settings']['estimator']
        if base.is_classifier(estimator_raw):
            estimator_final = estimator_raw
        else:
            est_algo_class = get_algo_class(estimator_raw['base_algorithm'])
            est_algo_kwargs = estimator_raw['keyword_arg_settings']
            est_algo_instance = est_algo_class(**est_algo_kwargs)
            estimator_final = est_algo_instance
        model_config['keyword_arg_settings']['estimator'] = estimator_final
        super(OutputCodeClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)
