
import pandas as pd

from feature.manipulator import Manipulator
from django.utils.text import slugify
from utils import load_clean_input_file_filepath

class Wrapper(Manipulator):

    def __init__(self, base_algorithm_class, model_config, project_settings, mode='algorithm'):
        kwargs = model_config['keyword_arg_settings']
        other_options = model_config['other_options']
        self.model_name = model_config['model_name']
        self.base_algorithm = base_algorithm_class(**kwargs)
        self.mode = mode
        if other_options.has_key('gen_output'):
            arg_val = other_options['gen_output']
            self.gen_output_flag = arg_val
        else:
            self.gen_output_flag = False
        if model_config['feature_settings']['select_before_eng']:
            manipulations = model_config['feature_settings']['feature_selection'] + model_config['feature_settings']['feature_engineering']
        else:
            manipulations = model_config['feature_settings']['feature_engineering'] + model_config['feature_settings']['feature_selection']
        super(Wrapper,self).__init__(model_config,project_settings,manipulations)

    def det_prior_init_feature_names_filepath(self, model_config):
        project_settings = self.project_settings
        filters = model_config['feature_settings']['feature_selection']
        transformers = model_config['feature_settings']['feature_engineering']
        mode = self.mode
        if model_config['feature_settings']['select_before_eng']:
            if mode == 'transformer':
                if len(filters) > 0:
                    prior_manipulator_name = filters[-1].keys()[0]
                    prior_manipulator_filepath = self._det_output_features_filepath(prior_manipulator_name)
                else:
                    prior_manipulator_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
            elif mode == 'filter':
                prior_manipulator_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
            else:
                raise NotImplementedError
        else:
            if mode == 'transformer':
                prior_manipulator_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
            elif mode == 'filter':
                if len(transformers) > 0:
                    prior_manipulator_name = transformers[-1].keys()[0]
                    prior_manipulator_filepath = self._det_output_features_filepath(prior_manipulator_name)
                else:
                    prior_manipulator_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
            else:
                raise NotImplementedError
        return prior_manipulator_filepath


    def fit(self,X,y):
        self.base_algorithm.fit(X,y)
        if hasattr(self.base_algorithm,'coef_'):
            setattr(self,'coef_',getattr(self.base_algorithm,'coef_'))
        if hasattr(self.base_algorithm,'feature_importances_'):
            setattr(self,'feature_importances_',getattr(self.base_algorithm,'feature_importances_'))
        if hasattr(self.base_algorithm,'estimators_'):
            setattr(self,'estimators_',getattr(self.base_algorithm,'estimators_'))
        return self

    def predict(self,X):
        y_pred =  self.base_algorithm.predict(X)
        self.y_pred = y_pred
        return y_pred

    def gen_output(self):
        y_pred = self.y_pred
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        pd.Series(y_pred).to_csv(artifact_dir + '/' + slugify(model_name) + '-validation-scores.txt',sep='\t',header=False,index=False)

    def set_features(self, working_features):
        setattr(self,'features', working_features)


