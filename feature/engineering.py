from __future__ import division

import importlib
import itertools

import pandas as pd
import numpy as np

from feature.base_transformers import InvOneHotEncoder, Interpolator, LeaveOneOutEncoder, Truncator, Deleter, Identity, \
    Sampler, Stacker, OOSPredictorEns, MetaModeler
from manipulator import ManipulatorChain, Manipulator
from algorithms.wrapper import Wrapper
from utils import flip_dict, load_inv_column_map, load_clean_input_file_filepath
from algorithms.algoutils import get_algo_class
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess

from django.utils.text import slugify

class TransformChain(ManipulatorChain):

    def __init__(self, starting_transformations, model_config, project_settings):
        updated_transformations = list()
        model_config['feature_settings']['order'] = 0
        engineering_module = importlib.import_module('feature.engineering')
        for transformation in starting_transformations:
            transformer_name = transformation.keys()[0].split('.')[-1:][0]
            transform_class = getattr(engineering_module, transformer_name)
            if transform_class.__bases__[0] == getattr(engineering_module,'TransformChain'):
                transform_chain_class = transform_class
                working_transformations = model_config['feature_settings']['feature_engineering']
                tc = transform_chain_class(working_transformations,model_config,project_settings)
                updated_transformations = tc.transformations
            else:
                transformer_name = transformation.keys()[0]
                transformer_class_name = transformer_name.split('.')[-1:][0]
                transform_class = getattr(engineering_module, transformer_class_name)
                transformer = transform_class(model_config, project_settings)
                model_config['feature_settings']['order'] += 1
                transformation[transformer_name]['initialized_manipulator'] = transformer
                updated_transformations = updated_transformations + [transformation]
        super(TransformChain,self).__init__(updated_transformations, model_config, project_settings)
        self.transformations = updated_transformations

    def transform(self,X_mat,y,dataset_name,fit_transform=False):
        X_mat_start = X_mat.copy()
        transformations = self.transformations
        if len(transformations) < 1:
            X_transform, y_transform = X_mat, y
        else:
            i = 1
            for d in transformations:
                transformer_name = d.keys()[0]
                transformer = d[transformer_name]['initialized_manipulator']
                transformer_class = transformer.__class__
                if fit_transform:
                    le = self.leak_enforcer
                    leak_exists = le.check_for_leak(X_mat)
                    leak_allowed = le.check_leak_allowed(transformer_name)
                    if leak_exists:
                        print "\t\tLeak found for manipulator: " + transformer_name
                        if leak_allowed:
                         print "\t\t\tLeak is allowed for " + transformer_name + ". CV Metrics will be invalid"
                        else:
                         X_mat, y = le.remove_leaking_indices(X_mat, y)
                    touch_cols = transformer.touch_indices
                    X_rel = X_mat.loc[:,touch_cols]
                    transformer.fit(X_rel, y)
                split_args = self._get_args(transformer_class, 'split')
                additional_args = filter(lambda x: x not in ['X_mat','y'], split_args)
                spkwargs = dict()
                for arg in additional_args:
                    spkwargs[arg] = eval(arg)
                X_touch, X_untouched, y_touch, y_untouched = transformer.split(X_mat, y,**spkwargs)
                transform_args = self._get_args(transformer_class, 'transform')
                additional_args = filter(lambda x: x not in ['X_touch','y_touch'], transform_args)
                tfkwargs = dict()
                for arg in additional_args:
                   tfkwargs[arg] = eval(arg)
                X_touched, y_touched = transformer.transform(X_touch, y_touch,**tfkwargs)
                combine_args = self._get_args(transformer_class, 'combine')
                additional_args = filter(lambda x: x not in ['X_touched','X_untouched','y_touched','y_untouched'],combine_args)
                cmkwargs = dict()
                for arg in additional_args:
                    cmkwargs[arg] = eval(arg)
                X_transform, y_transform = transformer.combine(X_touched, X_untouched, y_touched, y_untouched,**cmkwargs) #TODO: clean up this kwarg loading mess
                assert True not in pd.isnull(X_transform).any(1).value_counts() #TODO: pandas dependent
                if dataset_name == "train":
                    if transformer.store:
                        artifact_dir = self.artifact_dir
                        transformer.store_output(X_transform,output_dir=artifact_dir)
                X_mat, y = X_transform, y_transform
                i += 1
        return X_transform, y_transform

    def fit_transform(self,X_mat,y,dataset_name):
        return self.transform(X_mat,y,dataset_name,fit_transform=True)

class Transformer(Manipulator):

    def __init__(self, model_config, project_settings):
        manipulations = model_config['feature_settings']['feature_engineering']
        super(Transformer,self).__init__(model_config,project_settings,manipulations)
        transformer_name = self.manipulator_name
        transformer_settings = self.fetch_transform_settings(model_config,transformer_name)
        self.base_transformer = None
        if transformer_settings.has_key('kwargs'):
            kwargs = transformer_settings['kwargs']
        else:
            kwargs = dict()
        self.kwargs = kwargs
        if transformer_settings.has_key('store_output'):
            self.store = True
        else:
            self.store = False
        self.inclusion_patterns = transformer_settings['inclusion_patterns']

    def det_prior_init_feature_names_filepath(self, model_config):
        project_settings = self.project_settings
        if not model_config['feature_settings']['select_before_eng']:
            prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            filters = model_config['feature_settings']['feature_selection']
            num_filters = len(filters)
            if num_filters > 0:
                last_filter_name = filters[-1:][0].keys()[0]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(last_filter_name)
            else:
                prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings,'feature_names')
        return prior_manipulator_feature_names_filepath

    def configure_features(self):
        prior_features = self.load_prior_features()
        touch_cols, untouched_cols = self.det_relevant_cols(prior_features)
        self.touch_indices = touch_cols
        self.untouched_indices = untouched_cols
        touch_cols = self.touch_indices
        if not isinstance(self, HorizontalTransformer):
            new_features = self.gen_new_column_names(touch_cols, prior_features)
            new_feature_set = self.reindex(prior_features, new_features)
            self.features = new_feature_set
        else:
            self.features = prior_features
        self.output_features()

    def load_prior_features(self):
        prior_transform_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        prior_inv_col_map = load_inv_column_map(prior_transform_feature_names_filepath)
        prior_features = flip_dict(prior_inv_col_map)
        return prior_features

    def fetch_transform_settings(self,model_config, transformer_name):
        feature_eng_settings = model_config['feature_settings']['feature_engineering']
        for item in feature_eng_settings:
            if item.keys()[0] == transformer_name:
                transform_settings = item[transformer_name]
            else:
                pass
        return transform_settings

    def set_base_transformer(self,transformer_instance):
        self.base_transformer = transformer_instance

    def fit(self, X_mat, y, **kwargs):
        self.base_transformer.fit(X_mat, y)

    def det_relevant_cols(self, prior_features, exclusion_flag=False):
        assert getattr(self, 'inclusion_patterns') != None
        if hasattr(self,'exclusion_flag'):
            exclusion_flag = getattr(self,'exclusion_flag')
        include_columns = list()
        col_indices = prior_features.keys()
        col_names = prior_features.values()
        inclusion_patterns = self.inclusion_patterns
        inv_working_features = flip_dict(prior_features)
        if inclusion_patterns == ['All']:   #TODO: Handle this as a single string because I forget to make it a list in the yaml
            touch_indices = range(len(prior_features))
            untouched_indices = list()
        else:
            if type(inclusion_patterns) == dict:
                assert inclusion_patterns.keys() == ['All But']
                exclusion_flag = True
                inclusion_patterns = inclusion_patterns['All But']
            elif inclusion_patterns == "All Numeric":
                project_settings = self.project_settings
                assert project_settings.has_key('numeric_features')
                inclusion_patterns = project_settings['numeric_features']
            for pattern in inclusion_patterns:
                len_pat = len(pattern)
                pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
                include_columns = include_columns + pattern_begin_cols
            non_inter_include_columns = filter(lambda x: 'x%x' not in x, include_columns)
            touch_indices = [int(inv_working_features[col_name]) for col_name in non_inter_include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
        if exclusion_flag:
            return untouched_indices, touch_indices
        else:
            return touch_indices, untouched_indices

    def transform(self, X_touch, y_touch, **kwargs):
        """Transform Wrapper for Default (i.e. Vertical) Transformers"""
        X_touched = self.base_transformer.transform(X_touch, **kwargs)
        if type(X_touched) != pd.core.frame.DataFrame: #TODO: fix this when I move out of pandas
            rdf = pd.DataFrame(X_touched,index=X_touch.index)
        else:
            rdf = X_touched
        return rdf, y_touch

    def combine(self,X_touched,X_untouched,y_touched,y_untouched):
        """Pandas Dependent"""
        if X_untouched.shape[1] > 0:
            X_transform = pd.merge(X_untouched, X_touched, left_index=True, right_index=True)
            assert len(X_transform) == len(X_untouched) == len(X_touched)
        else:
            X_transform = X_touched
        features = self.features
        assert len(features) == X_transform.shape[1]
        X_transform.columns = range(len(features))
        if y_untouched is not None:
            print "this is mean to be a vertical transform. y_touched is not None which seems like a horizontal transform"
            raise Exception
        return X_transform, y_touched

    def store_output(self,X_mat,output_dir): #TODO: When I need to output filter, make this an abstract method in new ManipulatorChain class
        model_name = self.model_name
        transform_name = str(self.__class__).split('.')[-1:][0]
        output_file_name = slugify(model_name + '-' + transform_name)
        X_mat.to_csv(output_dir + '/' + output_file_name + '.txt', header=False, index=False, sep='\t')  # TODO: fix this when I move out of pandas

class HorizontalTransformer(Transformer):

    def __init__(self, model_config, project_settings):
        super(HorizontalTransformer,self).__init__(model_config, project_settings)

    def split(self, X_mat, y, dataset_name):
        """This is a horizontal splitting method"""
        if dataset_name == "train":
            X_mat_idx = X_mat.index.tolist()
            untouched_indices = self.untouched_indices
            touch_indices = self.touch_indices
            X_untouched = X_mat.loc[untouched_indices,:]
            y_untouched = pd.Series(y, index=X_mat_idx).loc[untouched_indices].tolist()
            X_touch = X_mat.loc[touch_indices,:]
            y_touch = pd.Series(y, index=X_mat_idx).loc[touch_indices].tolist()
            return X_touch, X_untouched, y_touch, y_untouched
        else:
            return None, X_mat, None, y


    def transform(self,X_touch,y_touch,dataset_name):
        if dataset_name == "train":
            X_touched, y_touched = self.base_transformer.transform(X_touch, y_touch)
            return X_touched, y_touched
        else:
            return X_touch, y_touch

    def combine(self,X_touched,X_untouched,y_touched,y_untouched,dataset_name):
        "This is a vertical combine instead of the default horizontal"
        if dataset_name == "train":
            X_transform = X_untouched.append(X_touched,ignore_index=True)
            y_transform = y_touched + y_untouched
            return X_transform, y_transform
        else:
            return X_untouched, y_untouched

class basis_expansion(Transformer):
    """
    yaml usage example:

    Models:
      <Model Name>:
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - basis_expansion:
                 inclusion_patterns:
                  - <feature_name>
                 kwargs:
                    include_bias: <bool>
                    interaction_only: <bool>
    """
    def __init__(self, model_config, project_settings):
        super(basis_expansion, self).__init__(model_config, project_settings)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))
        self.configure_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        tuples = list()
        for i in range(len(orig_tcol_idx)):
            b = orig_tcol_idx[i]
            rem_list = orig_tcol_idx[i:]
            t = [(working_features[b],working_features[r_i]) for r_i in rem_list]
            tuples = tuples + t
        stiched_feature_names = [self.stich(x) for x in tuples]
        orig_tcol_names = [working_features[idx] for idx in orig_tcol_idx]
        Xt_feat_names = orig_tcol_names + stiched_feature_names
        if getattr(self.base_transformer,'include_bias'):
            Xt_feat_names = ['bias'] + Xt_feat_names
        if getattr(self.base_transformer,'interaction_only'):
            Xt_feat_names = filter(lambda x: '**' not in x, Xt_feat_names)
        return Xt_feat_names


    def stich(self,tuple):
        if tuple[0] == tuple[1]:
            return "(" + tuple[0] + ')**2'
        else:
            return 'inter(' + tuple[0] + 'x%x' + tuple[1] + ')'

class interaction_terms(TransformChain):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - interaction_terms:
                  interactions: (examples below)
                    - "('bill_sep','hist_sep')"
                    - "('bill_sep','prepay_sep')"
    """

    def __init__(self, transformations, model_config, project_settings):
        Manipulator.__init__(self,model_config,project_settings,transformations) #TODO: figure out this troublesome line
        raw_interaction_strs = filter(lambda x: x.keys()[0] == 'interaction_terms',  transformations)[0]['interaction_terms']['interactions']
        compact_interactions = [eval(ris) for ris in raw_interaction_strs]
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_idx = transformer_names.index('interaction_terms')
        if t_idx == 0:
            if model_config['feature_settings']['select_before_eng']:
                print "Interaction terms are meant to be run before model selection methods. Turn off select_before_eng flag"
                raise Exception
            prior_transform_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            prior_transform = transformer_names[t_idx - 1]
            prior_transform_feature_names_filepath = self._det_output_features_filepath(prior_transform)
        inv_column_map = load_inv_column_map(prior_transform_feature_names_filepath)
        feature_names = inv_column_map.keys()
        i = 0
        for tuple in compact_interactions:
            t0 = tuple[0]
            t1 = tuple[1]
            # #TODO: Make this check work for categorical variables
            # if False in [x in feature_names for x in (t0,t1)]:
            #     print "Expected columns not present in data to do interaction. Check that they have not been transformed previously"
            #     raise Exception
            #     #TODO: move exception message inside Exception to be more informative.
            len_t0_str = len(t0)
            len_t1_str = len(t1)
            t0_col_vals = filter(lambda x: x[0:len_t0_str] == t0 , feature_names)
            t1_col_vals = filter(lambda x: x[0:len_t1_str] == t1 , feature_names)
            expanded_interactions = list(itertools.product(t0_col_vals ,t1_col_vals))
            for inter in expanded_interactions:
                transformation_dict = dict()
                transformation_dict['int' + str(i) + '.' + 'basis_expansion'] = {
                    'inclusion_patterns':[inter[0], inter[1]],
                    'kwargs' : {'include_bias': False, 'interaction_only': True}
                }
                expanded_transformations.append(transformation_dict)
                i += 1
        if t_idx == 0:
            transformations = expanded_transformations + transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(transformations) - 1:
            transformations = transformations[0:t_idx] + expanded_transformations + transformations[t_idx + 1:]
            exp_idx = len(transformations[0:t_idx]) + len(expanded_transformations)
        else:
            transformations = transformations[:-1] + expanded_transformations
            exp_idx = len(transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = transformations
        super(interaction_terms, self).__init__(transformations[:exp_idx], model_config, project_settings)

class normalize(Transformer):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - normalize:
                  inclusion_patterns:
                    - <pattern>
     (You can also specify 'All Numeric' to normalize all numeric columns, provided
     that you include a list of the numeric feature names in the project_settings yaml
     file in the src folder of the project directory) with the key 'numeric_features'
    """
    def __init__(self, model_config, project_settings):
        super(normalize, self).__init__(model_config, project_settings )
        self.set_base_transformer(Normalizer(**self.kwargs))
        self.configure_features()
        pass

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'norm(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class standard_scale(Transformer):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - standard_scale:
                  inclusion_patterns:
                    - <pattern>
     (You can also specify 'All Numeric' to scale all numeric columns, provided
     that you include a list of the numeric feature names in the project_settings yaml
     file in the src folder of the project directory) with the key 'numeric_features'
    """
    def __init__(self, model_config, project_settings):
        super(standard_scale, self).__init__(model_config, project_settings )
        self.set_base_transformer(StandardScaler(**self.kwargs))
        self.configure_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'scaled(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class pca(Transformer):
    """
    example yaml usage:
<model_name>
    feature_settings:
      feature_engineering:
        - pca:
            inclusion_patterns:
             - 'All'
            kwargs:
              n_components: <int>
    """

    def __init__(self, model_config, project_settings):
        super(pca, self).__init__(model_config, project_settings )
        self.set_base_transformer(PCA(**self.kwargs))
        self.configure_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        num_comps = self.base_transformer.n_components
        new_col_list = list()
        for i in range(num_comps):
            new_col_name = 'pc_' + str(i)
            new_col_list.append(new_col_name)
        return new_col_list

class loo_encoding(Transformer):
    """Example in models.yaml file:
        Models:
          Model Name:
           feature_settings:
             feature_engineering:
               - loo_encoding
                   inclusion_patterns
                     - <pattern>
    """
    def __init__(self, model_config, project_settings):
        super(loo_encoding, self).__init__(model_config, project_settings )
        self.set_base_transformer(LeaveOneOutEncoder(**self.kwargs))
        self.configure_features()

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch, dataset_name), y_touch

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        inclusion_patterns = self.inclusion_patterns
        assert len(inclusion_patterns) == 1
        pattern = inclusion_patterns[0]
        new_feature_name = 'loo(' + pattern  + ')'
        return [new_feature_name]

class stack(TransformChain):
    """Example in models.yaml file:
        Models:
          Model Name:
            base_algorithm: sklearn.linear_model.LinearRegression #This is combo method outlined in ESL, Chapter 8.8
            feature_engineering:
                - stack:
                    algorithms:
                    - <fully qualified algo> (i.e. sklearn.ensemble.RandomForestRegressor)
                        keyword_arg_settings:
                            random_state: 1234  #(this is illustrative)
                    - <another fully qualified algo>:
                        keyword_arg_settings:
                            ...
        # At bottom of feature engineering section, be sure to include only the derived algo columns
                - include_features:
                    inclusion_patterns: [<fully qualified algo 1>(str) + '.hat',
                                        <fully qualified algo 2>(str) + '.hat']
    """
    def __init__(self, starting_transformations, model_config, project_settings):
        Manipulator.__init__(self, model_config, project_settings,
                             starting_transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in starting_transformations]
        t_idx = transformer_names.index('stack') #TODO: Figure out whether next 3 lines uniquely identify stacking entry?
        i = 0
        stack_entry = filter(lambda x: x.keys()[0] == 'stack', starting_transformations)[0]['stack']
        algorithms = stack_entry['algorithms']
        derived_cols = list()
        for algorithm_dict in algorithms:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_stack'
            algorithm_name = algorithm_dict.keys()[0]
            assert len(algorithm_dict) == 1
            if len(derived_cols) == 0:
                inclusion_patterns = ['All']
            else:
                inclusion_patterns = {'All But': derived_cols}
            transformation_dict[expanded_transformer_name] = {
                'algorithm': algorithm_dict.keys()[0],
                'inclusion_patterns': inclusion_patterns,
                'keyword_arg_settings' : algorithm_dict[algorithm_name]['keyword_arg_settings']
            }
            derived_cols = derived_cols + [algorithm_name + '.hat']
            expanded_transformations.append(transformation_dict)
            i += 1
            pass
        if t_idx == 0:
            starting_transformations = expanded_transformations + starting_transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(starting_transformations) - 1:
            starting_transformations = starting_transformations[0:t_idx] + expanded_transformations + starting_transformations[t_idx + 1:]
            exp_idx = len(starting_transformations[0:t_idx]) + len(expanded_transformations)
        else:
            starting_transformations = starting_transformations[:-1] + expanded_transformations
            exp_idx = len(starting_transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = starting_transformations
        super(stack, self).__init__(starting_transformations[:exp_idx], model_config, project_settings)

class ind_stack(Transformer):
    """
    This is a utility transformation, used by stack TransformChain transformer
    """
    def __init__(self, model_config, project_settings):
        super(ind_stack, self).__init__(model_config, project_settings)
        ind_stack_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
        ['feature_engineering'])[0][self.manipulator_name] #TODO: Figure out if method fetch_transform_settings can do this?
        self.stacking_algorithm_name = ind_stack_entry['algorithm']
        self.configure_features()
        keyword_arg_settings = dict()
        if ind_stack_entry.has_key('keyword_arg_settings'):
            keyword_arg_settings = ind_stack_entry['keyword_arg_settings']
        self.set_base_transformer(Stacker(self.stacking_algorithm_name, keyword_arg_settings))

    def transform(self,X_touch,y_touch,dataset_name):
        stacker_instance = self.base_transformer
        col_loc = X_touch.columns.max() + 1
        ow = X_touch.shape[1]
        X_touch.loc[:,col_loc] = stacker_instance.transform(X_touch,dataset_name)
        assert X_touch.shape[1] > ow
        return X_touch, y_touch

    def gen_new_column_names(self, touch_indices, prior_features):
        stacking_algorithm_name = self.stacking_algorithm_name
        new_feature_name = stacking_algorithm_name + '.hat'
        feature_names = [prior_features[i] for i in touch_indices]
        return feature_names + [new_feature_name]

class kaggle_stack(TransformChain):
    """example in yaml.file
  Model Name:
    base_algorithm: algorithms.common.MetaModeler
    feature_settings:
      feature_engineering:
        - kaggle_stack:
            inclusion_patterns:
              - 'All'
            algorithms:
              - sklearn.ensemble.RandomForestRegressor:
                  keyword_arg_settings:
                    random_state: 1234
              - sklearn.ensemble.GradientBoostingRegressor:
                  keyword_arg_settings:
                    random_state: 1234
    """
    def __init__(self, starting_transformations, model_config, project_settings):
        Manipulator.__init__(self, model_config, project_settings,
                             starting_transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in starting_transformations]
        existing_include_feature_transformers = filter(lambda x: '.include_features' in x, transformer_names)
        if len(existing_include_feature_transformers) == 0:
            i = 0
        else:
            i = len(existing_include_feature_transformers)
        t_idx = transformer_names.index('kaggle_stack') #TODO: Is There better way to do this? What if you have a TC that appears more than once?
        kaggle_stack_entry = filter(lambda x: x.keys()[0] == 'kaggle_stack', starting_transformations)[0]['kaggle_stack']
        algorithms = kaggle_stack_entry['algorithms']
        derived_cols = [d.keys()[0] + '.hat' for d in algorithms]
        oos_predictor_ensemble_entry = dict()
        oos_predictor_ensemble_entry['oos_predictor_ensemble'] = {
            'algorithms': algorithms,
            'inclusion_patterns': ['All'],
        }
        expanded_transformations.append(oos_predictor_ensemble_entry)
        include_meta_features_transformer_settings = dict()
        include_meta_features_transformer_settings[str(i) + '.include_features'] = {
            'inclusion_patterns' : derived_cols
        }
        i += 1
        expanded_transformations.append(include_meta_features_transformer_settings)
        metamodel_transformer_settings = dict()
        metamodel_transformer_settings['metamodel'] = {
            'base_algorithm': 'sklearn.linear_model.LinearRegression',
            'keyword_arg_settings' : {},
            'inclusion_patterns': derived_cols
        }
        expanded_transformations.append(metamodel_transformer_settings)
        if t_idx == 0:
            starting_transformations = expanded_transformations + starting_transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(starting_transformations) - 1:
            starting_transformations = starting_transformations[0:t_idx] + expanded_transformations + starting_transformations[t_idx + 1:]
            exp_idx = len(starting_transformations[0:t_idx]) + len(expanded_transformations)
        else:
            starting_transformations = starting_transformations[:-1] + expanded_transformations
            exp_idx = len(starting_transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = starting_transformations
        super(kaggle_stack, self).__init__(starting_transformations[:exp_idx], model_config, project_settings)

class oos_predictor_ensemble(Transformer):
    """Utility class used by kaggle_stack

    (Refer to kaggle_stack TC constructor for more info if this transform is ever explicitly specified by user)
    """
    def __init__(self, model_config, project_settings):
        super(oos_predictor_ensemble, self).__init__(model_config, project_settings)
        oos_predictor_ensemble_settings = self.fetch_transform_settings(model_config, self.manipulator_name)
        algorithms = oos_predictor_ensemble_settings['algorithms']
        ens_algos = list()
        ens_algos_keyword_arg_dict = dict()
        for algo in algorithms:
            algo_name = algo.keys()[0]
            algo_keyword_arg_settings = algo[algo_name]['keyword_arg_settings']
            ens_algos.append(algo_name)
            ens_algos_keyword_arg_dict[algo_name] = algo_keyword_arg_settings
        self.ens_algos = ens_algos
        self.set_base_transformer(OOSPredictorEns(ens_algos, ens_algos_keyword_arg_dict))
        self.configure_features()

    def fit(self, X_touch, y_touch):
        model_config = self.model_config
        folds_info = { 'folds_map' : model_config['folds_map'], 'fold_i': model_config['fold_i']}
        oos_predictor_ensemble = self.base_transformer
        oos_predictor_ensemble.set_folds_info(folds_info)
        self.base_transformer.fit(X_touch,y_touch)

    def transform(self, X_touch, y_touch, dataset_name):
        oospredens_instance = self.base_transformer
        X_t = oospredens_instance.transform(X_touch, dataset_name)
        ow = X_touch.shape[1]
        X_t.columns = np.arange(ow,ow+X_t.shape[1],1)
        X_touched = pd.concat([X_touch,X_t],axis=1)
        assert X_touched.shape[1] == X_touch.shape[1] + X_t.shape[1]
        return X_touched, y_touch

    def gen_new_column_names(self, touch_indices, prior_features):
        ens_algos = self.ens_algos
        new_feature_names = list()
        for algo in ens_algos:
            new_feature_name = algo + '.hat'
            new_feature_names.append(new_feature_name)
        feature_names = [prior_features[i] for i in touch_indices]
        return feature_names + new_feature_names

class metamodel(Transformer):
    """Utility class used by kaggle_stack

    (Refer to kaggle_stack TC constructor for more info if this transform is ever explicitly specified by user)
    """
    def __init__(self, model_config, project_settings):
        super(metamodel, self).__init__(model_config, project_settings)
        metamodel_settings = self.fetch_transform_settings(model_config, self.manipulator_name)
        base_algorithm = metamodel_settings['base_algorithm']
        self.base_algorithm = base_algorithm
        keyword_arg_settings = metamodel_settings['keyword_arg_settings'] #TODO: Make defaults if no input provided
        self.set_base_transformer(MetaModeler(base_algorithm,keyword_arg_settings))
        self.configure_features()

    def transform(self, X_touch, y_touch, **kwargs):
        metamodel_transformer = self.base_transformer
        X_touched = pd.DataFrame(metamodel_transformer.fitted_base_algo.predict(X_touch),index=X_touch.index)
        return X_touched, y_touch

    def gen_new_column_names(self, touch_indices, prior_features):
        new_col_name = self.base_algorithm + '_metamodel'
        return [new_col_name]

class interpolate(Transformer):
    """Example in models.yaml file:
    Models:
      <model name>:
       feature_settings:
         feature_engineering:
           - interpolate
               inclusion_patterns
                 - <pattern>
                kwargs:
                    lowess: {}
                    interp1d: {}
    """
    def __init__(self, model_config, project_settings):
        super(interpolate, self).__init__(model_config, project_settings)
        self.set_base_transformer(None)
        self.configure_features()

    def fit(self, X_mat, y):
        kwargs = self.kwargs
        lowess_kwargs = kwargs['lowess']
        interp1d_kwargs = kwargs['interp1d']

        assert X_mat.shape[1] == 1 #For now, just one feature per transformer

        fitted_lowess = lowess(X_mat.iloc[:, 0], np.array(y), **lowess_kwargs)

        # unpack the lowess smoothed points to their values
        lowess_y = list(zip(*fitted_lowess))[0]
        lowess_x = list(zip(*fitted_lowess))[1]

        # run scipy's interpolation. There is also extrapolation I believe
        f = Interpolator(lowess_x, lowess_y, bounds_error=False, **interp1d_kwargs)

        self.set_base_transformer(f)

    def gen_new_column_names(self, touch_indices, prior_features):
        assert len(touch_indices) == 1
        touch_idx = touch_indices[0]
        old_feature_name = prior_features[touch_idx]
        new_feature_name = 'interp(' + old_feature_name + ')'
        return [new_feature_name]

class as_numeric(TransformChain):
    """Example in models.yaml file:
    Models:
      <model name>:
       feature_settings:
         feature_engineering:
           - as_numeric
               inclusion_patterns
                 - <pattern>
               val_maps:
                 - <pattern> : { str:val, str:val,...}
    """

    def __init__(self, starting_transformations, model_config, project_settings):
        Manipulator.__init__(self, model_config, project_settings,
                             transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_idx = transformer_names.index('as_numeric')
        i = 0
        as_numeric_entry = filter(lambda x: x.keys()[0] == 'as_numeric', transformations)[0]['as_numeric']
        inclusion_patterns = as_numeric_entry['inclusion_patterns']
        if as_numeric_entry.has_key('val_maps'):
            val_maps = as_numeric_entry['val_maps']
        else:
            val_maps = dict()
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_as_numeric'
            transformation_dict[expanded_transformer_name] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
            if val_maps.has_key(pattern):
                transformation_dict[expanded_transformer_name]['val_map'] = val_maps[pattern]
        if t_idx == 0:
            transformations = expanded_transformations + transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(transformations) - 1:
            transformations = transformations[0:t_idx] + expanded_transformations + transformations[t_idx + 1:]
            exp_idx = len(transformations[0:t_idx]) + len(expanded_transformations)
        else:
            transformations = transformations[:-1] + expanded_transformations
            exp_idx = len(transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = transformations
        super(as_numeric, self).__init__(transformations[:exp_idx], model_config, project_settings)

class ind_as_numeric(Transformer):
    """
    This is a utility transformation, used by as_numeric transformchain transformer
    """
    def __init__(self,model_config, project_settings):
        super(ind_as_numeric, self).__init__(model_config, project_settings)
        self.configure_features()
        ind_as_numeric_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
                        ['feature_engineering'])[0][self.manipulator_name]
        kwargs = dict()
        if ind_as_numeric_entry.has_key('val_map'):
            kwargs['val_map'] = ind_as_numeric_entry['val_map']
        self.set_base_transformer(InvOneHotEncoder(self.touch_indices, self.prior_features, **kwargs))


    def gen_new_column_names(self, touch_indices, prior_features):
        prior_feature_cols = [prior_features[ti] for ti in touch_indices]
        base_names = [pfc.split('_')[0] for pfc in prior_feature_cols]
        assert len(set(base_names)) == 1
        base_name = 'as_numeric(' + base_names[0] + ')'
        return [base_name]

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch,y_touch,dataset_name)

class sample(HorizontalTransformer):
    """Example in models.yaml file:
        Models:
          <Model Name>:
           feature_settings:
             feature_engineering:
               - sample:
                   inclusion_patterns:
                     - "All"
                   kwargs:
                      upsample: <boolean> (def: False):
    """
    def __init__(self,model_config,project_settings):
        super(sample, self).__init__(model_config, project_settings )
        self.set_base_transformer(Sampler(**self.kwargs))
        self.configure_features()

    def fit(self, X_mat, y):
        X_mat_idx = X_mat.index.tolist()
        y = pd.Series(y, index=X_mat_idx)
        min_idx = y[y == 1].index.tolist()
        maj_idx = y[y == 0].index.tolist()
        upsample = getattr(self.base_transformer,'upsample_flag')
        if upsample:
            touch_indices = min_idx
            untouched_indices = maj_idx
        else:
            touch_indices = maj_idx
            untouched_indices = min_idx
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        setattr(self.base_transformer,'touch_indices', touch_indices)
        setattr(self.base_transformer,'untouched_indices', untouched_indices)

    def gen_new_column_names(self, touch_indices, prior_features):
        return prior_features

class predict(Transformer):

    def __init__(self,model_config,project_settings):
        super(predict, self).__init__(model_config, project_settings)
        predict_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
                        ['feature_engineering'])[0][self.manipulator_name]
        base_algoritm = predict_entry['base_algorithm']
        base_algo_class = get_algo_class(base_algoritm)
        if not predict_entry.has_key('keyword_arg_settings'):
            predict_entry['keyword_arg_settings'] = dict()
        if not predict_entry.has_key('other_options'):
            predict_entry['other_options'] = dict()
        if not predict_entry.has_key('feature_settings'):
            predict_entry['feature_settings'] = { 'select_before_eng' : False,  #TODO: import these as defaults from
                                                  'feature_selection' : [],     #global_settings.yaml. Will require
                                                  'feature_engineering' : []    #some yaml parsing logic, must be somewhere in
                                                }                               #code already, to fill out config not def by user
        predict_entry['model_name'] = self.manipulator_name
        self.set_base_transformer(Wrapper(base_algo_class, predict_entry,project_settings))
        self.configure_features()

    def gen_new_column_names(self, touch_indices, prior_features):
        new_col_name = self.manipulator_name + '_hat'
        feature_names = [prior_features[i] for i in touch_indices]
        return feature_names + [new_col_name]

    def transform(self,X_touch,y_touch):
        base_algo_instance = self.base_transformer
        col_loc = X_touch.columns.max() + 1
        ow = X_touch.shape[1]
        X_touch.loc[:,col_loc] = base_algo_instance.predict(X_touch)
        assert X_touch.shape[1] > ow
        return X_touch, y_touch

class reset_data(Transformer):

    def __init__(self,model_config,project_settings):
        super(reset_data, self).__init__(model_config, project_settings )
        self.set_base_transformer(Identity(**self.kwargs))
        self.configure_features()

    def gen_new_column_names(self, touch_indices, prior_features):
        project_settings = self.project_settings
        clean_feature_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        clean_features = load_inv_column_map(clean_feature_filepath)
        return clean_features.keys()

    def transform(self,X_touch,y_touch,X_mat_start):
        return self.base_transformer.transform(X_mat_start), y_touch

class exclude_features(Transformer):
    """
    Example yaml usage:

    Models:
      <model name>:
        feature_settings:
          feature_selection:
            - exclude_features:
                inclusion_patterns:
                  - <matching pattern>
    """
    def __init__(self, model_config, project_settings):
        super(exclude_features, self).__init__(model_config, project_settings)
        self.configure_features()
        self.set_base_transformer(Deleter(**self.kwargs))

    def gen_new_column_names(self, touch_indices, prior_features):
        return list()

class include_features(Transformer):
    """
    Example yaml usage:

    Models:
      <model name>:
        feature_settings:
          feature_selection:
            - include_features:
                inclusion_patterns:
                  - <matching pattern>
    """
    def __init__(self, model_config, project_settings):
        super(include_features, self).__init__(model_config, project_settings)
        setattr(self,'exclusion_flag',True)
        self.configure_features()
        self.set_base_transformer(Deleter(**self.kwargs))

    def gen_new_column_names(self, touch_indices, prior_features):
        return list()

class drop_outliers(TransformChain):
    """TODO: Documentation?"""
    def __init__(self, transformations, model_config, project_settings):
        Manipulator.__init__(self, model_config, project_settings,
                             transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_idx = transformer_names.index('drop_outliers')
        i = 0
        drop_outliers_entry = filter(lambda x: x.keys()[0] == 'drop_outliers', transformations)[0]['drop_outliers']
        inclusion_patterns = drop_outliers_entry['inclusion_patterns']
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_drop_outliers'
            transformation_dict[expanded_transformer_name] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        if t_idx == 0:
            transformations = expanded_transformations + transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(transformations) - 1:
            transformations = transformations[0:t_idx] + expanded_transformations + transformations[t_idx + 1:]
            exp_idx = len(transformations[0:t_idx]) + len(expanded_transformations)
        else:
            transformations = transformations[:-1] + expanded_transformations
            exp_idx = len(transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = transformations
        super(drop_outliers, self).__init__(transformations[:exp_idx], model_config, project_settings)

class ind_drop_outliers(HorizontalTransformer):

    def __init__(self, model_config, project_settings):
        super(ind_drop_outliers, self).__init__(model_config, project_settings)
        self.set_base_transformer(Truncator(**self.kwargs))
        self.configure_features()

    def fit(self, X_col, y):
        X_mat_idx = X_col.index.tolist()
        y = pd.Series(y, index=X_mat_idx)
        assert X_col.shape[1] == 1
        ti = X_col.columns[0]
        truncator = self.base_transformer
        truncator.fit(X_col)
        X_mat_sub = X_col[(X_col.loc[:, ti] > truncator.lthrsh) & (X_col.loc[:, ti] < truncator.uthrsh)].copy()
        untouched_indices = list(X_mat_sub.index)
        touch_indices = list(set(X_mat_idx).difference(set(untouched_indices)))
        assert len(touch_indices) + len(untouched_indices) == len(X_col)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        #setattr(self.base_transformer, 'touch_indices', touch_indices)
        #setattr(self.base_transformer, 'untouched_indices', untouched_indices)

    # TODO: Delete this? Should HorizontalTransformers ever implement gen_new_column_names?
    # def gen_new_column_names(self, touch_indices, prior_features):
    #     relevant_features = [prior_features[ti] for ti in touch_indices]
    #     new_features = list()
    #     for rf in relevant_features:
    #         new_features.append('trunc(' + rf + ')')
    #     return new_features