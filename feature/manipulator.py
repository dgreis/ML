import pandas as pd
import os

from django.utils.text import slugify

from utils import find_data_dir, find_project_dir, load_working_file_filepath, load_inv_column_map


class Manipulator:

    def __init__(self, manipulations, model_config, project_settings,original_columns):

        num_manipulations = len(manipulations)
        if num_manipulations > 0:
            project_dir = find_project_dir(project_settings)
            artifact_dir = project_dir + '/model_artifacts'
            if not os.path.isdir(artifact_dir):
                os.makedirs(artifact_dir)
            self.artifact_dir = artifact_dir

        self.model_config = model_config
        self.project_settings = project_settings

        self.working_features = None

        if original_columns == False:
            feature_names_filepath = load_working_file_filepath(self.project_settings, 'feature_names')
        else:
            feature_names_filepath = self.load_clean_input_file_filepath('feature_names')
        inv_column_map = load_inv_column_map(feature_names_filepath)
        self.inv_column_map = inv_column_map

    def load_clean_input_file_filepath(self,data_ref):
        project_settings = self.project_settings
        cif_rel_filepath = project_settings['clean_input_files'][data_ref]
        data_dir = find_data_dir(project_settings)
        cif_abs_filepath = data_dir + '/' + cif_rel_filepath
        return cif_abs_filepath

    #def fit(self,X_train,y_train=None):
    #    pass

    def fit_transform(self,X_train,y_train=None):
        pass

    def transform(self,X_train):
        pass

    def _set_working_features(self, col_map):
        self.working_features = col_map

    def _output_features(self,feature_version):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_file_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        pd.Series(self.working_features).to_csv(artifact_dir + '/' + model_file_name_prefix + '-' + slugify(feature_version.replace('_','-')) + \
                                                '-features.txt',index=False,sep='\t')

    def _update_working_data_feature_names_ref(self, feature_version):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_file_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        project_settings = self.project_settings
        project_settings['working_files']['feature_names'] = artifact_dir + '/' + model_file_name_prefix + \
                                                             '-' + (feature_version.replace('_','-')) + '-features.txt'