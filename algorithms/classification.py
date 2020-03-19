
from utils import flip_dict, load_inv_column_map
from .wrapper import Wrapper
from .common import DecisionTree
from sklearn import tree

class DecisionTreeClassifier(DecisionTree):

    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = tree.DecisionTreeClassifier
        super(DecisionTreeClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

# class Bagged_Classifier(Wrapper):
#
#     #TODO: Implement this. This is currently a placeholder
#     def __init__(self, wrapper_id, base_algorithm_class, model_config='algorithm'):
#         base_estimator_name = kwargs['estimator']
#         base_estimator_class = None
#         super(Bagged_Classifier, self).__init__(,