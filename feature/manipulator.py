import pandas as pd
import os

from django.utils.text import slugify

from utils import find_data_dir, find_project_dir


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
            column_names_filepath = self.load_working_file_filepath('feature_names')
        else:
            column_names_filepath = self.load_clean_input_file_filepath('feature_names')
        inv_column_map = self.load_inv_column_map(column_names_filepath)
        self.inv_column_map = inv_column_map

    def load_inv_column_map(self, filepath):
        df = pd.read_csv(filepath, sep="\s+", engine='python', names=['col_name'])
        return pd.Series(df.index, index=df.col_name).to_dict()

    def load_working_file_filepath(self,data_ref):
        project_settings = self.project_settings
        assert project_settings.has_key('working_files')
        working_files = project_settings['working_files']
        filepath = working_files[data_ref]
        if self._is_fully_qualified_path(filepath):
            wf_abs_filepath = filepath
        else:
            data_dir = find_data_dir(project_settings)
            wf_abs_filepath = data_dir + '/' + filepath
        return wf_abs_filepath

    def load_clean_input_file_filepath(self,data_ref):
        project_settings = self.project_settings
        cif_rel_filepath = project_settings['clean_input_files'][data_ref]
        data_dir = find_data_dir(project_settings)
        cif_abs_filepath = data_dir + '/' + cif_rel_filepath
        return cif_abs_filepath

    def _is_fully_qualified_path(self,path):
        project_settings = self.project_settings
        repo_loc = project_settings['repo_loc']
        if repo_loc in path:
            return True
        else:
            return False

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