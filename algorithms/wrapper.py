
import pandas as pd

from feature.manipulator import Manipulator
from django.utils.text import slugify
from utils import load_clean_input_file_filepath, flip_dict

class Wrapper(Manipulator):

    def __init__(self, wrapper_id, base_algorithm_class, model_config, project_settings):
        kwargs = model_config['keyword_arg_settings']
        other_options = model_config['other_options']
        self.model_name = model_config['model_name']
        self.base_algorithm = base_algorithm_class(**kwargs)
        if 'gen_output' in other_options:
            arg_val = other_options['gen_output']
            self.gen_output_flag = arg_val
        else:
            self.gen_output_flag = False
        super(Wrapper, self).__init__(wrapper_id, model_config, project_settings)

    def fit(self,X,y):
        prior_features = self.load_prior_features()
        setattr(self, 'features', prior_features)
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

    def _det_prior_features_filepath(self,manipulator_name, manipulations, project_settings):
        if self.manipulator_name == 'final_algorithm':
            len_manipulations = len(manipulations)
            if len_manipulations == 0:
                prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings,'feature_names')
            else:
                manipulator_map = self.manipulator_map
                ord_manip_lookup = flip_dict(manipulator_map)
                pm_order = len_manipulations - 1
                prior_manipulator_name = ord_manip_lookup[pm_order]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(prior_manipulator_name)
            return prior_manipulator_feature_names_filepath
        else:
            return super(Wrapper, self)._det_prior_features_filepath(manipulator_name, manipulations, project_settings)


