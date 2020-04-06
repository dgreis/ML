
import pandas as pd
import importlib

from feature.manipulator import Manipulator
from sklearn import base
from django.utils.text import slugify
from utils import load_clean_input_file_filepath, flip_dict
from .algoutils import get_algo_class

class Wrapper(Manipulator):

    def __init__(self, wrapper_id, base_algorithm_class, model_config, project_settings):
        kwargs = model_config['keyword_arg_settings']
        self.model_name = model_config['model_name']
        if any(x in model_config['keyword_arg_settings'].keys() for x in ['estimator','estimators']):
            model_config = self._initialize_estimators(model_config)
        self.base_algorithm = base_algorithm_class(**kwargs)
        if len(model_config['output']) > 0:
            self.gen_output_flag = True
        else:
            self.gen_output_flag = False
        super(Wrapper, self).__init__(wrapper_id, model_config, project_settings)

    def _initialize_estimators(self, model_config):
        key = list(filter(lambda x: 'estimator' in x, model_config['keyword_arg_settings'].keys()))[0]
        if key == 'estimator':
            model_config = self._handle_single_estimator(model_config)
        else:
            model_config = self._handle_multiple_estimators(model_config)
        return model_config

    def _handle_single_estimator(self, model_config):
        estimator_raw = model_config['keyword_arg_settings']['estimator']
        if base.is_classifier(estimator_raw):
            return model_config
        else:
            est_algo_class = get_algo_class(estimator_raw['base_algorithm'])
            est_algo_kwargs = estimator_raw['keyword_arg_settings']
            est_algo_instance = est_algo_class(**est_algo_kwargs)
            estimator_final = est_algo_instance
            model_config['keyword_arg_settings']['estimator'] = estimator_final
            return model_config

    def _handle_multiple_estimators(self, model_config):
        estimators_raw = model_config['keyword_arg_settings']['estimators']
        if type(estimators_raw) == list:
            return model_config
        else:
            estimators_final = list()
            for k in estimators_raw:
                est_algo_class = get_algo_class(estimators_raw[k]['base_algorithm'])
                est_algo_kwargs = estimators_raw[k]['keyword_arg_settings']
                est_algo_instance = est_algo_class(**est_algo_kwargs)
                estimators_final.append((k, est_algo_instance))
            model_config['keyword_arg_settings']['estimators'] = estimators_final
            return model_config

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


