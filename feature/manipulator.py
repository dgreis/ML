import inspect
import pandas as pd
import os
import algorithms.wrapper

from django.utils.text import slugify
from utils import find_project_dir, flip_dict, load_clean_input_file_filepath


class Manipulator(object):

    def __init__(self, manipulator_id, model_config, project_settings):
        super(Manipulator,self).__init__()
        self.manipulator_name = manipulator_id #TODO: change all manipulator_name to id
        self.model_config = model_config
        self.project_settings = project_settings
        project_dir = find_project_dir(project_settings)
        artifact_dir = project_dir + '/model_artifacts'
        if not os.path.isdir(artifact_dir):
            os.makedirs(artifact_dir)
        self.artifact_dir = artifact_dir
        self.features = None
        self.validation_peeking = False
        manipulations = model_config['feature_settings']['manipulations']
        manipulator_names = [d.keys()[0] for d in manipulations]
        manipulator_map = dict(zip(manipulator_names, range(len(manipulator_names))))
        self.manipulator_map = manipulator_map
        if not issubclass(self.__class__, (ManipulatorChain, algorithms.wrapper.Wrapper)):
            m_order = manipulator_map[manipulator_id]
            if m_order == 0:
                prior_manipulator = None
                prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
            else:
                ord_manip_lookup = flip_dict(manipulator_map)
                prior_manipulator = ord_manip_lookup[m_order - 1]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(prior_manipulator)
            self.prior_manipulator_feature_names_filepath = prior_manipulator_feature_names_filepath

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

    def gen_new_column_names(self, touch_indices, prior_features):
        raise NotImplementedError

    def split(self, X_mat, y):
        untouched_indices = self.untouched_indices
        touch_indices = self.touch_indices
        X_untouched = X_mat.loc[:,untouched_indices]
        X_touch = X_mat.loc[:,touch_indices]
        return X_touch, X_untouched, y, None

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

    def __init__(self, manipulator_id, manipulations, model_config, project_settings):
        super(ManipulatorChain, self).__init__(manipulator_id, model_config, project_settings)
        self.leak_enforcer = None
        self.manipulations = manipulations

    def set_leak_enforcer(self,leak_enforcer):
        self.leak_enforcer = leak_enforcer

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