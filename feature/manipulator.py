import inspect

import pandas as pd
import os

from django.utils.text import slugify

from utils import find_project_dir, load_working_file_filepath, load_inv_column_map, \
    load_clean_input_file_filepath


class Manipulator(object):

    def __init__(self,model_config, project_settings):
        super(Manipulator,self).__init__()
        self.model_config = model_config
        self.project_settings = project_settings
        project_dir = find_project_dir(project_settings)
        artifact_dir = project_dir + '/model_artifacts'
        if not os.path.isdir(artifact_dir):
            os.makedirs(artifact_dir)
        self.artifact_dir = artifact_dir
        self.features = None

    def _det_output_features_filepath(self,manipulator_name):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        output_features_filepath = artifact_dir + '/' + model_name_prefix + '-' + slugify(manipulator_name.replace('_', '-')) + '-features.txt'
        return output_features_filepath

    def output_features(self):
        #class_name = str(self.__class__).split('.')[-1:][0]
        #TODO: fix this is, this is really bad
        manipulator_name = self.transformer_name
        if self.features is None:
            raise Exception
        output_features_filepath = self._det_output_features_filepath(manipulator_name)
        pd.Series(self.features).to_csv(output_features_filepath, index=False, sep='\t')


class ManipulatorChain(Manipulator):

    def __init__(self, manipulations, model_config, project_settings,original_columns):
        super(ManipulatorChain, self).__init__(model_config,project_settings)

        self.working_features = None

        if original_columns == False:
            feature_names_filepath = load_working_file_filepath(self.project_settings, 'feature_names')
        else:
            if model_config['feature_settings']['select_before_eng']:
                feature_names_filepath = artifact_dir + '/' + slugify(model_config['model_name']) +\
                                         '-selected-features-features.txt'
            else:
                feature_names_filepath = load_clean_input_file_filepath(self.project_settings, 'feature_names')
        inv_column_map = load_inv_column_map(feature_names_filepath)
        self.inv_column_map = inv_column_map
        self.original_columns = original_columns

    #def fit(self,X_train,y_train=None):
    #    pass

    def _pass_y_to_self(self, y):
        self.y = y

    def fit_transform(self,X_train,y_train=None):
        pass

    def transform(self,X_train):
        pass

    def _set_working_features(self, col_map):
        self.working_features = col_map

    def _update_working_data_feature_names_ref(self, feature_version):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_file_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        project_settings = self.project_settings
        project_settings['working_files']['feature_names'] = artifact_dir + '/' + model_file_name_prefix + \
                                                             '-' + (feature_version.replace('_','-')) + '-features.txt'

    def _get_args(self, class_, method):
        target = getattr(class_,method)
        args = getattr(inspect.getargspec(target),'args')
        return args[1:]