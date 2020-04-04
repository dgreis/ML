
from utils import flip_dict, load_inv_column_map
from .wrapper import Wrapper
from .common import DecisionTree
from sklearn import tree
from sklearn import ensemble
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

# class Bagged_Classifier(Wrapper):
#
#     #TODO: Implement this. This is currently a placeholder
#     def __init__(self, wrapper_id, base_algorithm_class, model_config='algorithm'):
#         base_estimator_name = kwargs['estimator']
#         base_estimator_class = None
#         super(Bagged_Classifier, self).__init__(,