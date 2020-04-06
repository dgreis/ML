
import pandas as pd
import importlib

from feature.manipulator import Manipulator
from django.utils.text import slugify
from utils import load_clean_input_file_filepath, flip_dict

class Wrapper(Manipulator):

    def __init__(self, wrapper_id, base_algorithm_class, model_config, project_settings):
        kwargs = model_config['keyword_arg_settings']
        self.model_name = model_config['model_name']
        self.base_algorithm = base_algorithm_class(**kwargs)
        if len(model_config['output']) > 0:
            self.gen_output_flag = True
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
        model_config = self.model_config
        output_options = model_config['output']
        if len(output_options) == 0:
            self.gen_default_output()
        else:
            wrapper_module_name = self.__module__
            wrapper_module = importlib.import_module(wrapper_module_name)
            wrapper_class = getattr(wrapper_module, self.__class__.__name__)
            for oo in output_options:
                if type(oo) == str:
                    method_name = oo
                    kwargs = {}
                elif type(oo) == dict:
                    assert len(oo) == 1
                    method_name = list(oo.keys())[0]
                    kwargs = oo[method_name]
                getattr(wrapper_class, method_name)(self,**kwargs)

    def gen_default_output(self):
        y_pred = self.y_pred
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        pd.Series(y_pred).to_csv(artifact_dir + '/' + slugify(model_name) + '-validation-scores.txt',sep='\t',header=False,index=False)

    def dump_design(self, data):
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        data.to_csv(artifact_dir + '/' + slugify(model_name) + '-design-matrix.txt', sep='\t', header=False, index=False)

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


