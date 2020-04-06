import pandas as pd
import math

from utils import flip_dict, load_inv_column_map
from .wrapper import Wrapper
from .common import DecisionTree
from django.utils.text import slugify
from sklearn import tree
from sklearn import ensemble
from sklearn import multiclass
from sklearn import base
from algorithms.algoutils import get_algo_class
from sklearn_gbmi import h_all_pairs

class DecisionTreeClassifier(DecisionTree):
    """Utility class to help common.DecisionTree know what base algo to use"""
    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = tree.DecisionTreeClassifier
        super(DecisionTreeClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

#TODO: Change all algo initialization to handle estimator (or multiple estimators) that require their own intialization first
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

class AugGradientBoostingClassifier(Wrapper):

    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = ensemble.GradientBoostingClassifier
        super(AugGradientBoostingClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

    def fit(self,X,y):
        self.X = X
        super().fit(X,y)

    def feature_importances(self):
        fitted_algo = self.base_algorithm
        fi = pd.Series(fitted_algo.feature_importances_)
        features = self.features
        feature_names = [features[i] for i in range(len(features))]
        fi.index = feature_names
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        fi.sort_values(ascending=False).to_csv(artifact_dir +'/' + slugify(model_name) + '-feature-importances.txt'
                  ,header=False
                  )

    def interaction_strengths(self,feature_importance_thresh):
        """
        Example Usage:
          Testing GBM Ints:
            base_algorithm: algorithms.classification.AugGradientBoostingClassifier
            output:
              - interaction_strengths:
                  feature_importance_thresh: 0.5 ( top x% if less than 1, num
                                                    to be considered if > 1 AND int)
        """
        fitted_algo = self.base_algorithm
        fi = pd.Series(fitted_algo.feature_importances_).sort_values(ascending=False)
        if feature_importance_thresh <= 1:
            incl_feat_idx = fi.head(math.ceil(feature_importance_thresh*len(fi))).index
        elif type(feature_importance_thresh) != int:
            raise Exception
        elif feature_importance_thresh > 1:
            incl_feat_idx = fi.head(feature_importance_thresh).index
        else:
            raise Exception
        X = self.X
        raw_output_dict = h_all_pairs(fitted_algo,X.loc[:,incl_feat_idx])
        features = self.features
        output_dict = dict()
        for o in raw_output_dict:
            key = tuple([features[i] for i in o])
            output_dict[key] = raw_output_dict[o]
        df = pd.DataFrame.from_dict(output_dict, orient='index',columns=['val'])
        df.sort_values('val',ascending=False,inplace=True)
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        df.to_csv(artifact_dir +'/' + slugify(model_name) + '-interaction-strengths.txt'
                  ,header=False
                  )