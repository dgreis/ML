from __future__ import division

import importlib
import itertools
import re

import pandas as pd
import numpy as np

from feature.base_transformers import InvOneHotEncoder, Interpolator, LeaveOneOutEncoder, Deleter, Identity, \
    Sampler, Stacker, OOSPredictorEns, MetaModeler, Imputer, Recoder, ExpressionEvaluator
from .manipulator import ManipulatorChain, Manipulator
from algorithms.wrapper import Wrapper
from utils import flip_dict, load_inv_column_map, load_clean_input_file_filepath
from algorithms.algoutils import get_algo_class
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import boxcox

from django.utils.text import slugify
from collections import OrderedDict

class Transformer(Manipulator):

    def __init__(self, transformer_id, model_config, project_settings):
        super(Transformer, self).__init__(transformer_id, model_config, project_settings)
        transformer_name = self.manipulator_name
        if issubclass(self.__class__, ManipulatorChain):
            pass
        else:
            transformer_settings = self.fetch_transform_settings(model_config,transformer_name)
            self.base_transformer = None
            if 'kwargs' in transformer_settings:
                kwargs = transformer_settings['kwargs']
            else:
                kwargs = dict()
            self.kwargs = kwargs
            if 'store_output' in transformer_settings:
                self.store = True
            else:
                self.store = False
            #self.inclusion_patterns = transformer_settings['inclusion_patterns']
            self.exclusion_flag = False

    def configure_features(self):
        prior_features = self.load_prior_features()
        touch_indices, untouched_indices = self.return_touch_untouched_indices(prior_features)
        try:
            assert len(touch_indices) > 0 #This check might be redundant with the one in the Transformer.fit() method
        except AssertionError:
            print("Warning: " + self.manipulator_name + " is about touch 0 relevant columns.")
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        if not isinstance(self, HorizontalTransformer):
            new_features = self.gen_new_column_names(touch_indices, prior_features)
            new_feature_set = self.reindex(prior_features, new_features)
            self.update_inclusion_patterns(prior_features)
            self.features = new_feature_set
        else:
            self.features = prior_features
        self.output_features()

    def update_inclusion_patterns(self,prior_features):
        inclusion_patterns = self.inclusion_patterns
        if inclusion_patterns in ['All', ['All']]:
            updated_inclusion_patterns = prior_features.values()
        else:
            if type(inclusion_patterns) == dict:
                assert inclusion_patterns.keys() == ['All But']
                updated_inclusion_patterns = inclusion_patterns['All But']
            elif inclusion_patterns == "All Numeric":
                model_config = self.model_config
                assert model_config.has_key('numeric_features')
                #Below filters to any numeric features that haven't already been filtered
                updated_inclusion_patterns = filter(lambda x: x in model_config['numeric_features'], prior_features.values())
                try:
                    assert len(updated_inclusion_patterns) > 0
                except AssertionError:
                    print (self.manipulator_name + " tries to get all numeric features but none exist at run-time")
                    raise Exception
            elif type(inclusion_patterns) == list:
                updated_inclusion_patterns = inclusion_patterns
            else:
                raise Exception
        self.inclusion_patterns = updated_inclusion_patterns

    def check_if_derived_column_is_numeric(self,derived_column):
        original_numeric_columns = self.project_settings['numeric_features']
        for onc in original_numeric_columns:
            if onc in derived_column:
                return True
            else:
                pass
        return False

    def get_base_column_name(self,derived_column):
        base_column_components = derived_column.split('(')[1:]
        if len(base_column_components) == 1:
            base_column_name = base_column_components[0].strip(')')
        else:
            base_column_name = ('(').join(base_column_components)[:-1]
        return base_column_name

    def fetch_transform_settings(self,model_config, transformer_name):
        feature_eng_settings = model_config['feature_settings']['manipulations']
        for item in feature_eng_settings:
            if list(item.keys())[0] == transformer_name:
                transform_settings = item[transformer_name]
            else:
                pass
        return transform_settings

    def set_base_transformer(self,transformer_instance):
        self.base_transformer = transformer_instance

    def fit(self, X_mat, y, **kwargs):
        prior_features = self.load_prior_features()
        inclusion_patterns = self.inclusion_patterns
        try:
            # Check that all columns due to be touched are in inclusion patterns
            assert pd.Series([pattern in prior_features.values() for pattern in inclusion_patterns ]).all()
        except AssertionError:
            print("inclusion patterns of " + self.manipulator_name + " don't exist in data at runtime. Please remove and re-run")
            raise Exception
        self.base_transformer.fit(X_mat, y)

    def has_prior_tag(self, tag_type):
        model_config = self.model_config
        manipulator_name = self.manipulator_name
        manipulator_map = self.manipulator_map
        pm_i = manipulator_map[manipulator_name] - 1
        manipulations = model_config['feature_settings']['manipulations']
        prior_manip_name = manipulations[pm_i].keys()[0]
        prior_manip_instance = manipulations[pm_i][prior_manip_name]['initialized_manipulator']
        tagger_module = importlib.import_module('feature.tagger')
        specific_tagger_class = getattr(tagger_module, tag_type)
        if issubclass(prior_manip_instance.__class__, specific_tagger_class):
            return True
        else:
            return False

    def set_inclusion_columns(self):
        model_config = self.model_config
        transformer_id = self.manipulator_name
        transform_settings = self.fetch_transform_settings(model_config, transformer_id)
        init_inclusion_patterns = transform_settings['inclusion_patterns']
        if init_inclusion_patterns in ['All',['All']]:
            inclusion_patterns = init_inclusion_patterns
        elif type(init_inclusion_patterns) == dict:
            assert init_inclusion_patterns.keys() == ['All But']
            setattr(self, 'exclusion_flag', True)
            inclusion_patterns = init_inclusion_patterns['All But']
        elif init_inclusion_patterns == "All Numeric":
            assert self.has_prior_tag('tag_numeric')
            inclusion_patterns = model_config['numeric_features']
        elif init_inclusion_patterns == "All Skewed":
            assert self.has_prior_tag('tag_skewed')
            inclusion_patterns = model_config['skewed_features']
        elif type(init_inclusion_patterns) == list:
            inclusion_patterns = init_inclusion_patterns
        else:
            raise NotImplementedError
        self.inclusion_patterns = inclusion_patterns

    def return_touch_untouched_indices(self, prior_features):
        assert getattr(self, 'inclusion_patterns') != None
        if hasattr(self,'exclusion_flag'):
            exclusion_flag = getattr(self,'exclusion_flag')
        inclusion_patterns = self.inclusion_patterns
        if inclusion_patterns in ['All',['All']]:
            touch_indices = range(len(prior_features))
            untouched_indices = list()
        else:
            include_columns = list()
            for pattern in inclusion_patterns:
                col_indices = prior_features.keys()
                col_names = prior_features.values()
                relevant_pattern_columns = self.det_relevant_columns(pattern, col_names)
                include_columns = include_columns + list(relevant_pattern_columns)
            inv_working_features = flip_dict(prior_features)
            touch_indices = [int(inv_working_features[col_name]) for col_name in include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
        if self.exclusion_flag:
            return untouched_indices, touch_indices
        else:
            return touch_indices, untouched_indices

    def det_relevant_columns(self, pattern, col_names):
        pattern_col = filter(lambda x: x == pattern, col_names)
        return pattern_col

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
            print("this is mean to be a vertical transform. y_touched is not None which seems like a horizontal transform")
            raise Exception
        return X_transform, y_touched

    def store_output(self,X_mat,output_dir): #TODO: When I need to output filter, make this an abstract method in new ManipulatorChain class
        model_name = self.model_name
        transform_name = str(self.__class__).split('.')[-1:][0]
        output_file_name = slugify(model_name + '-' + transform_name)
        X_mat.to_csv(output_dir + '/' + output_file_name + '.txt', header=False, index=False, sep='\t')  # TODO: fix this when I move out of pandas

class HorizontalTransformer(Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(HorizontalTransformer, self).__init__(transformer_id, model_config, project_settings)

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
            X_transform = X_untouched.append(X_touched,ignore_index=False)
            assert len(X_transform) > 0
            assert not pd.Series(X_transform.index).duplicated().any()
            y_touched_ser = pd.Series(y_touched, index=X_touched.index)
            y_untouched_ser = pd.Series(y_untouched, index=X_untouched.index)
            y_transform_ser = y_touched_ser.append(y_untouched_ser, ignore_index=False)
            assert not pd.Series(y_transform_ser.index).duplicated().any()
            y_transform = y_transform_ser.tolist()
            return X_transform, y_transform
        else:
            return X_untouched, y_untouched

    def update_touch_indices(self,X_mat):
        touch_indices = self.touch_indices
        untouched_indices = filter(lambda x: x not in touch_indices, X_mat.index.tolist())
        self.untouched_indices = untouched_indices

class Cleaner(Transformer):

    #TODO: Figure out multiple inheritance here to set validation_peeking attribute once for all Cleaners
    def __init(self):
        pass

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(basis_expansion, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))
        #self.configure_features()

    def gen_new_column_names(self, touch_indices, prior_features):
        tuples = list()
        for i in range(len(touch_indices)):
            b = touch_indices[i]
            rem_list = touch_indices[i:]
            t = [(prior_features[b], prior_features[r_i]) for r_i in rem_list]
            tuples = tuples + t
        stiched_feature_names = [self.stich(x) for x in tuples]
        orig_tcol_names = [prior_features[idx] for idx in touch_indices]
        Xt_feat_names = orig_tcol_names + stiched_feature_names
        if getattr(self.base_transformer,'include_bias'):
            Xt_feat_names = [self.manipulator_name + '.bias'] + Xt_feat_names
        if getattr(self.base_transformer,'interaction_only'):
            Xt_feat_names = filter(lambda x: '**' not in x, Xt_feat_names)
        return Xt_feat_names

    def fit(self,X,y):
        assert X.shape[1] < 3 #If more than this, something is most likely wrong
        self.base_transformer.fit(X,y)

    def stich(self,tuple):
        if tuple[0] == tuple[1]:
            return "(" + tuple[0] + ')**2'
        #else:
        #    return 'inter(' + tuple[0] + 'x%x' + tuple[1] + ')'
        else:
            raise Exception

class ind_operator(Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_operator, self).__init__(transformer_id, model_config, project_settings)
        transformer_settings = self.fetch_transform_settings(model_config, transformer_id)
        equals = transformer_settings['equals']
        if 'keep_cols' in transformer_settings:
            keep_cols = transformer_settings['keep_cols']
        else:
            keep_cols = True
        self.keep_cols = keep_cols
        self.equals = equals
        self.operator = transformer_settings['operator']
        self.set_base_transformer(None)

    def update_model_config(self, transformer_id, new_config, model_config):
        manipulations = model_config['feature_settings']['manipulations']
        manipulator_map = dict(zip(range(len(manipulations)),[list(x.keys())[0] for x in manipulations]))
        inv_man_map = flip_dict(manipulator_map)
        t_idx = inv_man_map[transformer_id]
        model_config['feature_settings']['manipulations'][t_idx][transformer_id] = new_config
        return model_config

    def fit(self, X_mat, y, **kwargs):
        raise NotImplementedError

    def gen_new_column_names(self, touch_indices, prior_features):
        keep_cols = self.keep_cols
        equals = self.equals
        if not keep_cols:
            new_features =  [equals]
        else:
            new_features = [prior_features[i] for i in touch_indices] + [equals]
        return new_features

class ind_primal_op(ind_operator):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_primal_op, self).__init__(transformer_id, model_config, project_settings)

    def fit(self, X_mat, y, **kwargs):
        inclusion_patterns = self.inclusion_patterns
        touch_indices = self.touch_indices
        assert len(inclusion_patterns) == len(touch_indices) == 2
        operator = self.operator
        eval_str = 'X_touch.loc[:, ' + str(touch_indices[0]) + ']'
        rel_col = touch_indices[1]
        eval_str = eval_str + ' ' + operator + ' ' + 'X_touch.loc[:, ' +\
                            str(rel_col) + ']'
        keep_cols = self.keep_cols
        self.set_base_transformer(ExpressionEvaluator(eval_str,keep_cols))

class ind_trig_op(ind_operator):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_trig_op, self).__init__(transformer_id, model_config, project_settings)

    def fit(self, X_mat, y, **kwargs):
        inclusion_patterns = self.inclusion_patterns
        touch_indices = self.touch_indices
        assert len(inclusion_patterns) == len(touch_indices) == 1
        operator = self.operator
        eval_str = 'np.' + operator + '(X_touch.loc[:, ' + str(touch_indices[0]) + '])'
        keep_cols = self.keep_cols
        self.set_base_transformer(ExpressionEvaluator(eval_str,keep_cols))

class ind_constant_op(ind_operator):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_constant_op, self).__init__(transformer_id, model_config, project_settings)

    def fit(self, X_mat, y, **kwargs):
        inclusion_patterns = self.inclusion_patterns
        touch_indices = self.touch_indices
        assert len(inclusion_patterns) == len(touch_indices) == 1
        operator = self.operator
        eval_str =  'X_touch.loc[:, ' + str(touch_indices[0]) + ']' + operator
        keep_cols = self.keep_cols
        self.set_base_transformer(ExpressionEvaluator(eval_str,keep_cols))

class ind_interaction_terms(basis_expansion):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_interaction_terms, self).__init__(transformer_id, model_config, project_settings)

    def configure_features(self):
        prior_features = self.load_prior_features()
        inclusion_patterns = self.inclusion_patterns
        matches = list()
        for ip in inclusion_patterns:
            matches_ip = [ip in pf for pf in prior_features.values()]
            matches = matches + matches_ip
        try:
            assert pd.Series(matches).any()
        except AssertionError:
             print("Specified interaction: (" + ', '.join(inclusion_patterns) + ') not available at run-time. Please remove from yaml file')
             raise Exception
        base_features = dict()
        col_indices = list()
        for key in prior_features:
            base_feature = prior_features[key]
            while ')' in base_feature:
                base_feature = self.get_base_column_name(base_feature)
            base_features[key] = base_feature
            col_indices = col_indices + [key]
        assert len(inclusion_patterns) == 2
        t0,t1 = inclusion_patterns[0],inclusion_patterns[1]
        len_t0_str = len(t0)
        len_t1_str = len(t1)
        base_feature_names = base_features.values()
        t0_col_vals = filter(lambda x: (x[0:len_t0_str] == t0) and ('x%x' not in x), base_feature_names)
        t1_col_vals = filter(lambda x: (x[0:len_t1_str] == t1) and ('x%x' not in x), base_feature_names)
        expanded_interactions = list(itertools.product(t0_col_vals, t1_col_vals))
        self.expanded_interactions = expanded_interactions
        relevant_base_cols = list(set([item for sublist in expanded_interactions for item in sublist])) #flattened list
        col_to_ind_map = flip_dict(base_features)
        touch_indices = [col_to_ind_map[rbc] for rbc in relevant_base_cols]
        untouched_indices = list(set(col_indices).difference(set(touch_indices)))
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        new_features = self.gen_new_column_names(touch_indices, prior_features)
        new_feature_set = self.reindex(prior_features, new_features)
        #self.update_inclusion_patterns(prior_features)
        self.inclusion_patterns = [prior_features[i] for i in touch_indices]
        self.features = new_feature_set
        self.output_features()

    def fit(self,X,y):
        expanded_interactions = self.expanded_interactions
        prior_features = self.load_prior_features() #TODO: figure out a point to just make this a member rather than load
        col_to_ind_map = flip_dict(prior_features)
        pairwise_int_transformer_dict = OrderedDict()
        for inter in expanded_interactions:
            ind_tuple = (col_to_ind_map[inter[0]], col_to_ind_map[inter[1]])
            X_int = X.loc[:,ind_tuple]
            base_transformer = self.base_transformer
            base_transformer.fit(X_int,y)
            pairwise_int_transformer_dict[ind_tuple] = base_transformer
            self.set_base_transformer(PolynomialFeatures(**self.kwargs))
        self.pairwise_int_transformer_dict = pairwise_int_transformer_dict

    def transform(self, X_touch, y_touch, **kwargs): #TODO: is there a way I could remove unused y_touch's from these signatures?
        pairwise_int_transformer_dict = self.pairwise_int_transformer_dict
        pairwise_ints = pairwise_int_transformer_dict.keys()
        feat_col_dict = { 'f0' : pd.DataFrame(), 'f1' : pd.DataFrame()}
        intr_df = pd.DataFrame()
        #X_touched = pd.DataFrame() #pandas dependent, as usual
        i = 0
        for pwi in pairwise_ints:
            transformer_i = pairwise_int_transformer_dict[pwi]
            X_t = transformer_i.transform(X_touch.loc[:, pwi])
            X_touched_pwi = pd.DataFrame(X_t, index=X_touch.index)
            for n in [0,1]:
                if pwi[n] not in feat_col_dict['f' + str(n)].columns:
                    col_idx = pwi[n]
                    feat_col_dict['f' + str(n)].loc[:,col_idx] = X_touched_pwi.loc[:,n]
                else:
                    pass
            intr_df.loc[:,i] = X_touched_pwi.loc[:,2]
            i += 1
        f0_df, f1_df = feat_col_dict['f0'], feat_col_dict['f1']
        X_touched = pd.concat([f0_df, f1_df, intr_df],axis=1)
        assert X_touched.shape[1] == f0_df.shape[1] * f1_df.shape[1] + f0_df.shape[1] + f1_df.shape[1]
        return X_touched, y_touch

    def gen_new_column_names(self, touch_indices, prior_features):
        expanded_interactions = self.expanded_interactions
        feat_col_dict = {'f0_names' : list(), 'f1_names': list()}
        intr_names = list()
        for inter in expanded_interactions:
            for n in [0,1]:
                col_list = feat_col_dict['f' + str(n) + '_names']
                if inter[n] not in col_list:
                    col_list = col_list + [inter[n]]
                    feat_col_dict['f' + str(n) + '_names'] = col_list
                else:
                    pass
            inter_term_name = 'inter(' + inter[0] + 'x%x' + inter[1] + ')'
            intr_names = intr_names + [inter_term_name]
        f0_names, f1_names = feat_col_dict['f0_names'], feat_col_dict['f1_names']
        new_features = f0_names + f1_names + intr_names
        return new_features

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
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(normalize, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(Normalizer(**self.kwargs))
        #self.configure_features()

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
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(standard_scale, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(StandardScaler(**self.kwargs))
        #self.configure_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'scaled(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class ind_box_cox_transform(Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_box_cox_transform, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(None)

    def fit(self, X_mat, y, **kwargs):
        pass

    def transform(self, X_touch, y_touch):
        touch_indices = self.touch_indices
        kwargs = self.kwargs
        if kwargs.has_key('lmbda'):
            l = kwargs['lmbda']
        else:
            l = 0.15
        X_touched = X_touch.apply(lambda x: boxcox(x + 0.000001, lmbda=l), axis = 0)
        return X_touched, y_touch

    def gen_new_column_names(self, touch_indices, prior_features):
        orig_col_names = [prior_features[i] for i in touch_indices]
        new_col_names = ['boxCox('+ cn + ')' for cn in orig_col_names]
        return new_col_names

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

    def __init__(self, transformer_id, model_config, project_settings):
        super(pca, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(PCA(**self.kwargs))
        #self.configure_features()

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(loo_encoding, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(LeaveOneOutEncoder(**self.kwargs))
        #self.configure_features()

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch, dataset_name), y_touch

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        inclusion_patterns = self.inclusion_patterns
        assert len(inclusion_patterns) == 1
        pattern = inclusion_patterns[0]
        new_feature_name = 'loo(' + pattern  + ')'
        return [new_feature_name]

class ind_stack(Transformer):
    """
    This is a utility transformation, used by stack TransformChain transformer
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_stack, self).__init__(transformer_id, model_config, project_settings)
        ind_stack_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
        ['feature_engineering'])[0][self.manipulator_name] #TODO: Figure out if method fetch_transform_settings can do this?
        self.stacking_algorithm_name = ind_stack_entry['algorithm']
        #self.configure_features()
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

class oos_predictor_ensemble(Transformer):
    """Utility class used by kaggle_stack

    (Refer to kaggle_stack TC constructor for more info if this transform is ever explicitly specified by user)
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(oos_predictor_ensemble, self).__init__(transformer_id, model_config, project_settings)
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
        validation_peeking = oos_predictor_ensemble_settings['validation_peeking']
        #TODO: Some kind of type checking for all inputs?
        assert validation_peeking in [True, False]
        self.validation_peeking = validation_peeking
        self.set_base_transformer(OOSPredictorEns(ens_algos, ens_algos_keyword_arg_dict, self.validation_peeking))
        #self.configure_features()

    def fit(self, X_touch, y_touch):
        model_config = self.model_config
        folds_info = { 'folds_map' : model_config['folds_map'], 'fold_i': model_config['fold_i']}
        oos_predictor_ensemble = self.base_transformer
        oos_predictor_ensemble.set_folds_info(folds_info)
        self.base_transformer.fit(X_touch,y_touch)

    def transform(self, X_touch, y_touch, dataset_name):
        oospredens_instance = self.base_transformer
        X_touched, y_touched = oospredens_instance.transform(X_touch, y_touch, dataset_name)
        ow = X_touch.shape[1]
        X_touched.columns = np.arange(ow,ow+X_touched.shape[1],1)
        X_touched = pd.concat([X_touch,X_touched],axis=1,join='inner')
        num_ensemble_algos = len(oospredens_instance.ens_algos)
        assert X_touched.shape[1] == X_touch.shape[1] + num_ensemble_algos
        return X_touched, y_touched

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(metamodel, self).__init__(transformer_id, model_config, project_settings)
        metamodel_settings = self.fetch_transform_settings(model_config, self.manipulator_name)
        base_algorithm = metamodel_settings['base_algorithm']
        self.base_algorithm = base_algorithm
        keyword_arg_settings = metamodel_settings['keyword_arg_settings'] #TODO: Make defaults if no input provided
        self.set_base_transformer(MetaModeler(base_algorithm,keyword_arg_settings))
        #self.configure_features()

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(interpolate, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(None)
        #self.configure_features()

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

class ind_as_numeric(Transformer):
    """
    This is a utility transformation, used by as_numeric transformchain transformer
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_as_numeric, self).__init__(transformer_id, model_config, project_settings)
        transformer_settings = self.fetch_transform_settings(model_config,transformer_id)
        val_map = transformer_settings['val_map']
        self.val_map = val_map
        self.set_base_transformer(None)

    def fit(self,X,y):
        val_map = self.val_map
        touch_indices = self.touch_indices
        prior_features = self.load_prior_features()
        self.set_base_transformer(InvOneHotEncoder(touch_indices, prior_features, val_map))

    def gen_new_column_names(self, touch_indices, prior_features):
        prior_feature_cols = [prior_features[ti] for ti in touch_indices]
        base_names = [pfc.split('_')[0] for pfc in prior_feature_cols]
        assert len(set(base_names)) == 1
        base_name = 'asNumeric(' + base_names[0] + ')'
        return [base_name]

    def det_relevant_columns(self, pattern, col_names):
        #TODO: Refactor/rethink this method/inheritance strategy. This is duplicated from recode
        len_pat = len(pattern)
        pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
        non_inter_begin_cols = filter(lambda x: 'x%x' not in x, pattern_begin_cols)
        return non_inter_begin_cols

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch,dataset_name), y_touch

class ind_encode(Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_encode, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(OneHotEncoder(sparse=False))

    def fit(self, X_mat, y, **kwargs):
        #TODO: if this becomes some kind of pre-processing transformer, I think here I'll need
        #to save the val -> int index map here
        pass

    def transform(self, X_touch, y_touch, **kwargs):
        touch_indices = self.touch_indices
        assert len(touch_indices) == 1
        touch_idx = touch_indices[0]
        input_array = np.array(list(X_touch.loc[:, touch_idx])).reshape(-1, 1)
        enc = self.base_transformer
        X_touched = pd.DataFrame(enc.fit_transform(input_array),index=X_touch.index)
        return X_touched, y_touch

    def configure_features(self, X_dev):
        prior_features = self.load_prior_features()
        touch_indices, untouched_indices = self.return_touch_untouched_indices(prior_features)
        try:
            assert len(touch_indices) > 0 #This check might be redundant with the one in the Transformer.fit() method
        except AssertionError:
            print("Warning: " + self.manipulator_name + " is about touch 0 relevant columns.")
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        assert len(touch_indices) == 1
        touch_idx = touch_indices[0]
        cat_vals = X_dev.loc[:,touch_idx].value_counts().index.values.tolist()
        base_col = prior_features[touch_idx]
        new_features = sorted([base_col + '_' + str(x) for x in cat_vals])
        new_feature_set = self.reindex(prior_features, new_features)
        self.update_inclusion_patterns(prior_features)
        self.features = new_feature_set
        self.output_features()

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(sample, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(Sampler(**self.kwargs))
        #self.configure_features()

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

    def __init__(self, transformer_id, model_config, project_settings):
        super(predict, self).__init__(transformer_id, model_config, project_settings)
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
        self.set_base_transformer(Wrapper(transformer_id, base_algo_class, predict_entry, project_settings))
        #self.configure_features()

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

    def __init__(self, transformer_id, model_config, project_settings):
        super(reset_data, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(Identity(**self.kwargs))
        #self.configure_features()

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
    def __init__(self, transformer_id, model_config, project_settings):
        super(exclude_features, self).__init__(transformer_id, model_config, project_settings)
        #self.configure_features()
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
    def __init__(self, transformer_id, model_config, project_settings):
        super(include_features, self).__init__(transformer_id, model_config, project_settings)
        setattr(self,'exclusion_flag',True)
        #self.configure_features()
        self.set_base_transformer(Deleter(**self.kwargs))

    def gen_new_column_names(self, touch_indices, prior_features):
        return list()

#TODO: Figure out whether this can also be part of data pre-processing
class ind_drop_outliers(HorizontalTransformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_drop_outliers, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(Deleter(**self.kwargs))
        self.lower_thresh = None
        self.upper_thresh = None

    def fit(self, X_col, y):
        assert X_col.shape[1] == 1
        #implementing 1.5xIQR rule for now
        desc = X_col.describe()
        IQR = (desc.loc['75%'] - desc.loc['25%']).values[0]
        assert not pd.isnull(IQR)
        if IQR == 0:
            upper_thresh = np.nan
            lower_thresh = np.nan
        else:
            upper_thresh = desc.loc['75%'].values[0] + 1.5*IQR
            lower_thresh = desc.loc['25%'].values[0] - 1.5*IQR
        if pd.isnull(lower_thresh):
            X_sub = X_col
            print("\t\t" + self.inclusion_patterns[0] + " drop outliers has an IQR = 0. No observations will be dropped")
        else:
            X_sub = X_col[(X_col.iloc[:, 0] > lower_thresh) & (X_col.iloc[:, 0] < upper_thresh)]
        untouched_indices = list(X_sub.index)
        touch_indices = list(set(X_col.index).difference(set(untouched_indices)))
        assert len(touch_indices) + len(untouched_indices) == len(X_col)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices

class truncate(HorizontalTransformer):
    """
    truncate differs from drop_outliers in that it requires specific thresholds to operate.
    drop_outliers operates according to an overall strategy, which is why it can be applied
    to multiple columns at once.
    sample yaml:
    ...
    feature_settings:
        manipulations:
            - truncate:
                inclusion_patterns:
                    - <col_name> (Should be only one column)
                thresholds: (<lower>,<upper>)
    """
    def __init__(self, transformer_id, model_config, project_settings):
        super(truncate, self).__init__(transformer_id, model_config, project_settings)
        transformer_settings = self.fetch_transform_settings(model_config,transformer_id)
        inclusion_patterns = transformer_settings['inclusion_patterns']
        col_thresh_map = dict()
        for d in inclusion_patterns:
            k,v = d.items()[0]
            col_thresh_map[k] = v
        self.col_thresh_map = col_thresh_map
        self.set_base_transformer(Deleter(**self.kwargs))

    def fit(self, X_cols, y):
        col_thresh_map = self.col_thresh_map
        project_settings = self.project_settings
        target_variable = project_settings['target_variable']
        prior_features = self.load_prior_features()
        inv_col_map = flip_dict(prior_features)
        conditional_clauses = list()
        for col,thresholds in col_thresh_map.items():
            lower_thresh, upper_thresh = thresholds
            if col == target_variable:
                assert len(X_cols) == len(y)
                X_cols.loc[:, 'y'] = y
                cc = "(X_cols.loc[:,'y'] > " + str(lower_thresh) + ") & " \
                     "(X_cols.loc[:,'y'] < " + str(upper_thresh) + ")"
            else:
                cc = "(X_cols.loc[:," + str(inv_col_map[col]) + "] > " + str(lower_thresh) + ") & " \
                     "(X_cols.loc[:," + str(inv_col_map[col]) + "] < " + str(upper_thresh )+ ")"
            conditional_clauses = conditional_clauses + [cc]
        conditional_statement = " | ".join(conditional_clauses)
        X_sub = X_cols[eval(conditional_statement)]
        untouched_indices = list(X_sub.index)
        touch_indices = list(set(X_cols.index).difference(set(untouched_indices)))
        assert len(touch_indices) + len(untouched_indices) == len(X_cols)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices

    def return_touch_untouched_indices(self, prior_features, exclusion_flag=False):
        #TODO: Work out a more sensible way to handle inclusion_patterns across manipulators
        assert getattr(self, 'inclusion_patterns') != None
        if hasattr(self,'exclusion_flag'):
            exclusion_flag = getattr(self,'exclusion_flag')
        include_columns = list()
        col_indices = prior_features.keys()
        col_names = prior_features.values()
        inclusion_patterns = [d.keys()[0] for d in self.inclusion_patterns]
        inv_working_features = flip_dict(prior_features)
        if inclusion_patterns in ['All',['All']]:
            raise Exception
        else:
            if type(inclusion_patterns) == dict:
                assert inclusion_patterns.keys() == ['All But']
                raise Exception
            elif inclusion_patterns == "All Numeric":
                raise Exception
            for pattern in inclusion_patterns:
                relevant_pattern_columns = self.det_relevant_columns(pattern, col_names)
                include_columns = include_columns + relevant_pattern_columns
            touch_indices = [int(inv_working_features[col_name]) for col_name in include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
        if exclusion_flag:
            return untouched_indices, touch_indices
        else:
            return touch_indices, untouched_indices

class delete_obs(HorizontalTransformer, Cleaner):

    def __init__(self, transformer_id, model_config, project_settings):
        #TODO: Implement exclusion_patterns, i.e. 'All But'. Or maybe it already works. I don't know. Figure it out
        super(delete_obs, self).__init__(transformer_id, model_config, project_settings)
        self.validation_peeking = True #TODO: Figure out how to make multiple inheritance work so Cleaner is only place this needs to be stated
        self.set_base_transformer(Deleter())

    def fit(self, X, y):
        #Now check for any missing columns
        if pd.isnull(X).any(0).any():
            touch_indices = {k:v for k,v in pd.isnull(X).any(1).to_dict().items() if v == True}.keys()
            untouched_indices = {k:v for k,v in pd.isnull(X).any(1).to_dict().items() if v == False}.keys()
        else:
            touch_indices = []
            untouched_indices = X.index
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        assert len(touch_indices) != len(X)
        #if you run afoul of assertion above, it's probably because you have data missing
        #in every row.

class ind_delete_var(Cleaner, Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_delete_var, self).__init__(transformer_id, model_config, project_settings)
        self.set_base_transformer(Deleter(**self.kwargs))

    def fit(self, X, y):
        pass

    def gen_new_column_names(self, touch_indices, prior_features):
        return []

    def det_relevant_columns(self, pattern, col_names):
        #TODO: Refactor/rethink this method/inheritance strategy. This is duplicated from recode
        len_pat = len(pattern)
        pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
        non_inter_begin_cols = filter(lambda x: 'x%x' not in x, pattern_begin_cols)
        return non_inter_begin_cols

class ind_impute_var(Cleaner, Transformer):

    def __init__(self, transformer_id, model_config, project_settings):
        super(ind_impute_var, self).__init__(transformer_id, model_config, project_settings)
        self.validation_peeking = True
        transformer_settings = self.fetch_transform_settings(model_config, transformer_id)
        replace_with = transformer_settings['replace_with']
        self.set_base_transformer(Imputer(replace_with))

    def gen_new_column_names(self, touch_indices, prior_features):
        assert len(touch_indices) == 1
        return [prior_features[touch_indices[0]]]

class recode(Cleaner, Transformer):
    #TODO: Documentation?
    #TODO: test to see what happens when recoder encounters previously unseen value. This does happen in house_prices data
    def __init__(self, transformer_id, model_config, project_settings):
        super(recode, self).__init__(transformer_id, model_config, project_settings)
        self.validation_peeking = True
        tranformer_settings = self.fetch_transform_settings(model_config, transformer_id)
        desc_val_map = tranformer_settings['val_map']
        inclusion_patterns = tranformer_settings['inclusion_patterns']
        assert len(inclusion_patterns) == 1
        #self.inclusion_patterns == inclusion_patterns
        self.desc_val_map = desc_val_map
        self.set_base_transformer(None)

    def fit(self, X,y):
        prior_features = self.load_prior_features()
        touch_indices = self.touch_indices
        inclusion_pattern = self.inclusion_patterns[0]
        prior_feature_names = prior_features.values()
        cat_col_names = self.det_relevant_columns(inclusion_pattern, prior_feature_names)
        inv_col_map = flip_dict(prior_features)
        cat_col_indices = [inv_col_map[ccn] for ccn in cat_col_names]
        ind_min = min(cat_col_indices)
        assert cat_col_indices == touch_indices
        assert (cat_col_indices == np.arange(ind_min, ind_min + len(cat_col_indices), 1)).all()
        self.fitted_indices = cat_col_indices
        desc_val_map = self.desc_val_map
        for suffix in desc_val_map:
            assert inclusion_pattern + '_' + suffix in cat_col_names
        new_col_names = self.gen_new_column_names(touch_indices, prior_features)
        new_col_suffixes = [x.split('_')[1] for x in new_col_names]
        lookup = dict(zip(new_col_suffixes,range(len(new_col_suffixes))))
        val_map = dict()
        for k,v in desc_val_map.items():
            if v == '<Mode>':
                mode_idx = X.loc[:,touch_indices].sum(0).idxmax() - ind_min
                v = flip_dict(lookup)[mode_idx]
            val_map[inv_col_map[inclusion_pattern + '_' + k]-ind_min] = lookup[v]
        self.set_base_transformer(Recoder(val_map))

    def det_relevant_columns(self, pattern, col_names):
        len_pat = len(pattern)
        pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
        non_inter_begin_cols = filter(lambda x: 'x%x' not in x, pattern_begin_cols)
        return non_inter_begin_cols

    def gen_new_column_names(self, touch_indices, prior_features):
        desc_val_map = self.desc_val_map
        relevant_prior_features = {k: v for k, v in prior_features.items() if k in touch_indices}
        assert len(relevant_prior_features) == len(touch_indices)
        num_orig_cols = len(relevant_prior_features)
        min_col = min(relevant_prior_features.keys())
        new_col_names = list()
        for i in np.arange(min_col,min_col + num_orig_cols, 1):
            col_name = relevant_prior_features[i]
            col_prefix, col_suffix = col_name.split('_')
            if col_suffix in desc_val_map:
                new_suffix = desc_val_map[col_suffix]
                if new_suffix == '<Mode>':
                    continue
                else:
                    new_col_name = col_prefix + '_' + new_suffix
            else:
                new_col_name = col_name
            if not new_col_name in new_col_names:
                new_col_names = new_col_names + [new_col_name]
        return new_col_names

    def transform(self, X_touch, y_touch, **kwargs):
        fitted_indices = self.fitted_indices
        #Try to make sure there are no new values of cat variable, though this isn't foolproof.
        #TODO: figure out how to handle new values of cat variable in unseen data
        assert (fitted_indices == X_touch.columns).all()
        return super(recode, self).transform(X_touch, y_touch, **kwargs)

