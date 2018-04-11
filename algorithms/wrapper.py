import os

import pandas as pd

from utils import find_project_dir, load_working_file_filepath, load_inv_column_map

class Wrapper(object):

    def __init__(self, base_algorithm_class, model_config, project_settings):
        kwargs = model_config['keyword_arg_settings']
        other_options = model_config['other_options']
        self.model_name = model_config['model_name']
        self.base_algorithm = base_algorithm_class(**kwargs)
        if other_options.has_key('gen_output'):
            arg_val = other_options['gen_output']
            self.gen_output_flag = arg_val
            if self.gen_output_flag:
                project_dir = find_project_dir(project_settings)
                artifact_dir = project_dir + '/model_artifacts'
                if not os.path.isdir(artifact_dir):
                    os.makedirs(artifact_dir)
                self.artifact_dir = artifact_dir
                feature_names_filepath = load_working_file_filepath(project_settings,'feature_names')
                inv_column_map = load_inv_column_map(feature_names_filepath)
                self.inv_column_map = inv_column_map
        else:
            self.gen_output_flag = False

    def fit(self,X,y):
        self.base_algorithm.fit(X,y)
        if hasattr(self.base_algorithm,'coef_'):
            setattr(self,'coef_',getattr(self.base_algorithm,'coef_'))
        if hasattr(self.base_algorithm,'feature_importances_'):
            setattr(self,'feature_importances_',getattr(self.base_algorithm,'feature_importances_'))
        return self

    def predict(self,X):
        return self.base_algorithm.predict(X)

    def gen_output(self):
        raise NotImplementedError