import inspect
import pandas as pd
import os
import importlib

from django.utils.text import slugify
from utils import find_project_dir, flip_dict, load_clean_input_file_filepath, load_inv_column_map

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
        manipulator_names = [list(d.keys())[0] for d in manipulations]
        manipulator_map = dict(zip(manipulator_names, range(len(manipulator_names))))
        self.manipulator_map = manipulator_map
        if not issubclass(self.__class__, ManipulatorChain):
            prior_manipulator_feature_names_filepath = self._det_prior_features_filepath(manipulator_id, manipulations,
                                                                                         project_settings)
            self.prior_manipulator_feature_names_filepath = prior_manipulator_feature_names_filepath

    def _det_output_features_filepath(self,manipulator_name):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        output_features_filepath = artifact_dir + '/' + model_name_prefix + '-' + slugify(manipulator_name.replace('_', '-')) + '-features.txt'
        return output_features_filepath

    def _det_prior_features_filepath(self,manipulator_name, manipulations, project_settings):
        assert len(manipulations) != 0
        manipulator_map = self.manipulator_map
        manipulator_name = self.manipulator_name
        m_order = manipulator_map[manipulator_name]
        tagger_module = importlib.import_module('feature.tagger')
        counter = m_order
        ord_man_map = flip_dict(manipulator_map)
        while counter > 0:
            cand_man_name = ord_man_map[counter - 1]
            init_man = list(filter(lambda x: list(x.keys())[0] == cand_man_name, manipulations))[0][cand_man_name]['initialized_manipulator']
            if issubclass(init_man.__class__, getattr(tagger_module, 'Tagger')):
                counter = counter - 1
            else:
                prior_man_name = cand_man_name
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(prior_man_name)
                return prior_manipulator_feature_names_filepath
        prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        return prior_manipulator_feature_names_filepath

    def fetch_manipulator_settings(self, model_config):
        manipulator_map = self.manipulator_map
        manipulator_name = self.manipulator_name
        t_idx = manipulator_map[manipulator_name]
        manipulations = model_config['feature_settings']['manipulations']
        manipulator_settings = manipulations[t_idx][manipulator_name]
        return manipulator_settings

    def output_features(self):
        manipulator_name = self.manipulator_name
        if self.features is None:
            raise Exception
        output_features_filepath = self._det_output_features_filepath(manipulator_name)
        pd.Series(self.features).to_csv(output_features_filepath, index=False, header=False, sep='\t')

    def gen_new_column_names(self, touch_indices, prior_features):
        raise NotImplementedError

    def split(self, X_mat, y):
        untouched_indices = self.untouched_indices
        touch_indices = self.touch_indices
        X_untouched = X_mat.loc[:,X_mat.columns.intersection(untouched_indices)]
        X_touch = X_mat.loc[:,X_mat.columns.intersection(touch_indices)]
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

    def load_prior_features(self):
        prior_transform_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        prior_inv_col_map = load_inv_column_map(prior_transform_feature_names_filepath)
        prior_features = flip_dict(prior_inv_col_map)
        return prior_features

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

    def load_prior_features(self):
        raise NotImplementedError
    
    def _get_args(self, class_, method):
        target = getattr(class_,method)
        args = getattr(inspect.getargspec(target),'args')
        return args[1:]