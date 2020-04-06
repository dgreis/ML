import pandas as pd
import math

from utils import flip_dict, load_inv_column_map
from .wrapper import Wrapper
from .common import DecisionTree
from django.utils.text import slugify
from sklearn import tree
from sklearn import ensemble
from sklearn_gbmi import h_all_pairs

class DecisionTreeClassifier(DecisionTree):
    """Utility class to help common.DecisionTree know what base algo to use"""
    def __init__(self, wrapper_id, model_config, project_settings):
        base_algo_class = tree.DecisionTreeClassifier
        super(DecisionTreeClassifier, self).__init__(wrapper_id, base_algo_class, model_config, project_settings)

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