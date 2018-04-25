import inspect

import pandas as pd
import os

from django.utils.text import slugify

from utils import find_project_dir


class Manipulator(object):

    def __init__(self,model_config, project_settings,manipulations):
        super(Manipulator,self).__init__()
        self.model_config = model_config
        self.project_settings = project_settings
        project_dir = find_project_dir(project_settings)
        artifact_dir = project_dir + '/model_artifacts'
        if not os.path.isdir(artifact_dir):
            os.makedirs(artifact_dir)
        self.artifact_dir = artifact_dir
        self.features = None
        if len(manipulations) > 0:
            manipulator_names = [d.keys()[0] for d in manipulations]
            order = model_config['feature_settings']['order']
            if order <= 0:
                prior_manipulator_feature_names_filepath = self.det_prior_feature_names_filepath(model_config)
            else:
                prior_transform = manipulator_names[order-1]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(prior_transform)
            self.prior_manipulator_feature_names_filepath = prior_manipulator_feature_names_filepath
            manipulator_name = manipulator_names[order]
            self.manipulator_name = manipulator_name
        else:
            pass

    def det_prior_feature_names_filepath(self,model_config):
        raise NotImplementedError

    def _det_output_features_filepath(self,manipulator_name):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        output_features_filepath = artifact_dir + '/' + model_name_prefix + '-' + slugify(manipulator_name.replace('_', '-')) + '-features.txt'
        return output_features_filepath

    def output_features(self):
        manipulator_name = self.manipulator_name
        if self.features is None:
            raise Exception
        output_features_filepath = self._det_output_features_filepath(manipulator_name)
        pd.Series(self.features).to_csv(output_features_filepath, index=False, sep='\t')

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        raise NotImplementedError

    def split(self, X_mat, y_mat):
        untouched_indices = self.untouched_indices
        touch_indices = self.touch_indices
        X_untouched = X_mat.loc[:,untouched_indices]
        X_touch = X_mat.loc[:,touch_indices]
        return X_touch, X_untouched, None, y_mat

    def reindex(self, prior_features, new_features=list()):
        untouched_indices = self.untouched_indices
        new_col_map = dict()
        ni = 0
        for oi in untouched_indices:
            col_name = prior_features[oi]
            new_col_map[ni] = col_name
            ni += 1
        reindexed_col_idx = list()
        for feature in new_features:
            new_col_map[ni] = feature
            reindexed_col_idx.append(ni)
            ni += 1
        assert ni == len(new_features) + len(untouched_indices)
        return new_col_map


class ManipulatorChain(Manipulator):

    def __init__(self, manipulations, model_config, project_settings,original_columns):
        super(ManipulatorChain, self).__init__(model_config,project_settings, manipulations)

    def fit(self,X_mat,y):
        raise NotImplementedError

    def fit_transform(self,X_mat,y):
        raise NotImplementedError

    def transform(self,X_mat,y):
        raise NotImplementedError
    
    def _get_args(self, class_, method):
        target = getattr(class_,method)
        args = getattr(inspect.getargspec(target),'args')
        return args[1:]