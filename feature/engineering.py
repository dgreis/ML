from __future__ import division

import importlib
import itertools
import copy
import os

import pandas as pd

from manipulator import ManipulatorChain, Manipulator
from utils import flip_dict, find_project_dir, load_inv_column_map, load_clean_input_file_filepath
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from django.utils.text import slugify

class TransformChain(ManipulatorChain):

    def __init__(self,starting_transformations, model_config, project_settings,original_columns=False):
        updated_transformations = list()
        model_config['feature_settings']['t_order'] = 0
        engineering_module = importlib.import_module('feature.engineering')
        for transformation in starting_transformations:
            engineering_module = importlib.import_module('feature.engineering')
            transformer_name = transformation.keys()[0].split('.')[-1:][0]
            transform_class = getattr(engineering_module, transformer_name)
            if transform_class.__bases__[0] == getattr(engineering_module,'TransformChain'):
                transform_chain_class = transform_class
                tc = transform_chain_class(starting_transformations,model_config,project_settings,original_columns)
                updated_transformations = updated_transformations + tc.transformations
            else:
                transformer_name = transformation.keys()[0]
                transformer_class_name = transformer_name.split('.')[-1:][0]
                transform_class = getattr(engineering_module, transformer_class_name)
                transformer = transform_class(model_config, project_settings)
                transformation[transformer_name]['initialized_transformer'] = transformer
                updated_transformations = updated_transformations + [transformation]
                #expanded_model_config = copy.deepcopy(model_config)
            #model_config['feature_settings']['feature_engineering'] = updated_transformations
        super(TransformChain,self).__init__(updated_transformations, model_config, project_settings,original_columns)
        self.transformations = updated_transformations

    def transform(self,X_mat,y_mat,dataset_name="Train"):
        log_prefix = "[" + dataset_name + "]"
        #engineering_module = importlib.import_module('feature.engineering')
        transformations = self.transformations
        model_config = self.model_config
        project_settings = self.project_settings
        self._pass_y_to_self(y_mat)
        i = 1
        if len(transformations) < 1:
            X_transform = X_mat
        else:
            for d in transformations:
                #working_features = self.working_features
                #if working_features is None:
                #    inv_column_map = self.inv_column_map
                #    working_features = flip_dict(inv_column_map)
                transformer_name = d.keys()[0]
                #transformer_class_name = transformer_name.split('.')[-1:][0]
                print "\t" + log_prefix + " Performing feature engineering (" + str(i) + '/' + str(
                    len(transformations)) + "): " + transformer_name
                #transform_class = getattr(engineering_module, transformer_class_name)
                #transformer = transform_class(model_config, project_settings)
                transformer = d[transformer_name]['initialized_transformer']
                transformer.fit(X_mat,y_mat)
                X_touch, X_untouched, y_touch, y_untouched = transformer.split(X_mat,y_mat)
                X_touched, y_touched = transformer.transform(X_touch,y_touch)
                #X_touch, X_untouched = transformer.split(X_mat, working_features)
                #fit_transform_args = self._get_args(transform_class, 'fit_transform')
                #additional_args = filter(lambda x: x not in ['X_touch','working_features'], fit_transform_args)
                #kwargs = dict()
                #for arg in additional_args:
                #    kwargs[arg] = getattr(self, arg)
                #X_touched, new_feat_names = transformer.fit_transform(X_touch, working_features,**kwargs)
                X_transform = transformer.combine(X_touched, X_untouched, y_touched, y_untouched)
                #self._set_working_features(updated_col_map)
                if dataset_name == "Train":
                    #self._output_features(transformer_name)
                    #transformer.output_features()
                    #self._update_working_data_feature_names_ref(slugify(transformer_name))
                    if transformer.store:
                        artifact_dir = self.artifact_dir
                        transformer.store_output(X_transform,output_dir=artifact_dir)
                X_mat = X_transform
                i += 1
        return X_transform


class Transformer(Manipulator):

    def __init__(self, model_config, project_settings):
        super(Transformer,self).__init__(model_config,project_settings)
        self.base_transformer = None
        transformations = model_config['feature_settings']['feature_engineering']
        transformer_names = [d.keys()[0] for d in transformations]
        t_order = model_config['feature_settings']['t_order']
        transformer_name = transformer_names[t_order]
        self.transformer_name = transformer_name
        transformer_settings = self.fetch_transform_settings(model_config,transformer_name)
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
        if t_order == 0:
            prior_transform_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            prior_transform = transformer_names[t_order-1]
            prior_transform_feature_names_filepath = self._det_output_features_filepath(prior_transform)
        self.prior_transform_feature_names_filepath = prior_transform_feature_names_filepath

    def tbd(self):
        prior_transform_feature_names_filepath = self.prior_transform_feature_names_filepath
        model_config = self.model_config
        prior_inv_col_map = load_inv_column_map(prior_transform_feature_names_filepath)
        prior_features = flip_dict(prior_inv_col_map)
        touch_indices, untouched_indices = self.determine_split_indices(prior_features)
        new_features = self.gen_new_column_names(touch_indices, prior_features)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        new_feature_set = self.reindex(prior_features,new_features)
        self.features = new_feature_set
        self.output_features()
        model_config['feature_settings']['t_order'] += 1

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

    def fit(self,X_mat,y_mat):
        pass

    def determine_split_indices(self, prior_features):
        #TODO: Better name this fn?
        assert getattr(self, 'inclusion_patterns') != None
        include_columns = list()
        col_indices = prior_features.keys()
        col_names = prior_features.values()
        inclusion_patterns = self.inclusion_patterns
        inv_working_features = flip_dict(prior_features)
        if inclusion_patterns == ['All']:
            touch_indices = range(len(prior_features))
            untouched_indices = list()
        else:
            for pattern in inclusion_patterns:
                len_pat = len(pattern)
                pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
                include_columns = include_columns + pattern_begin_cols
            non_inter_include_columns = filter(lambda x: 'x%x' not in x, include_columns)
            touch_indices = [int(inv_working_features[col_name]) for col_name in non_inter_include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
            touch_indices = touch_indices
            untouched_indices = untouched_indices
        return touch_indices, untouched_indices

    def split(self, X_mat, y):
        untouched_indices = self.untouched_indices
        touch_indices = self.touch_indices
        X_untouched = X_mat.loc[:,untouched_indices]
        y_untouched = [y[i] for i in untouched_indices]
        X_touch = X_mat.loc[:,touch_indices]
        y_touch = [y[i] for i in touch_indices]
        return X_touch, X_untouched, y_touch, y_untouched

    def transform(self, X_touch, y_touch, **kwargs):
        X_touched = self.base_transformer.fit_transform(X_touch,**kwargs)
        if type(X_touched) != pd.core.frame.DataFrame: #TODO: fix this when I move out of pandas
            rdf = pd.DataFrame(X_touched)
        else:
            rdf = X_touched
        return rdf, y_touch

    def combine(self,X_touched,X_untouched,y_touched,y_untouched):
        """Pandas Dependent"""
        if len(X_untouched) > 0:
            X_transform = pd.merge(X_untouched, X_touched, left_index=True, right_index=True)
            assert len(X_transform) == len(X_untouched) == len(X_touched)
        else:
            X_transform = X_touched
        features = self.features
        assert len(features) == X_transform.shape[1]
        X_transform.columns = range(len(features))
        return X_transform

    def store_output(self,X_mat,output_dir): #TODO: When I need to output filter, make this an abstract method in new ManipulatorChain class
        model_name = self.model_name
        transform_name = str(self.__class__).split('.')[-1:][0]
        output_file_name = slugify(model_name + '-' + transform_name)
        X_mat.to_csv(output_dir + '/' + output_file_name + '.txt', header=False, index=False, sep='\t')  # TODO: fix this when I move out of pandas

    def reindex(self, prior_features, new_features):
        untouched_indices = self.untouched_indices
        new_col_map = dict()
        ni = 0
        for oi in untouched_indices:
            col_name = prior_features[oi]
            new_col_map[ni] = col_name
            ni += 1
        reindexed_col_idx = list()
        for feature in new_features:
            new_col_map[ni] = feature
            reindexed_col_idx.append(ni)
            ni += 1
        assert ni == len(new_features) + len(untouched_indices)
        return new_col_map

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        raise NotImplementedError

class basis_expansion(Transformer):

    def __init__(self, model_config, project_settings):
        super(basis_expansion, self).__init__(model_config, project_settings)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))
        self.tbd()

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
            return tuple[0] + '**2'
        else:
            return tuple[0] + 'x%x' + tuple[1]

class interaction_terms(TransformChain):

    def __init__(self,transformations, model_config, project_settings,original_columns):
        Manipulator.__init__(self,model_config,project_settings)
        raw_interaction_strs = filter(lambda x: x.keys()[0] == 'interaction_terms',  transformations)[0]['interaction_terms']['interactions']
        compact_interactions = [eval(ris) for ris in raw_interaction_strs]
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_order = transformer_names.index('interaction_terms')
        if t_order == 0:
            prior_transform_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            prior_transform = transformer_names[t_order - 1]
            prior_transform_feature_names_filepath = self._det_output_features_filepath(prior_transform)
        inv_column_map = load_inv_column_map(prior_transform_feature_names_filepath)
        feature_names = inv_column_map.keys()
        i = 0
        for tuple in compact_interactions:
            t0 = tuple[0]
            t1 = tuple[1]
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
        transformations.pop()
        transformations = transformations + expanded_transformations
        model_config['feature_settings']['feature_engineering'] = transformations
        super(interaction_terms, self).__init__(transformations, model_config, project_settings,original_columns)

class normalize(Transformer):

    def __init__(self, model_config, project_settings):
        super(normalize, self).__init__(model_config, project_settings )
        self.set_base_transformer(Normalizer(**self.kwargs))
        self.tbd()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'norm(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names


class standard_scale(Transformer):

    def __init__(self, model_config, project_settings):
        super(standard_scale, self).__init__(model_config, project_settings )
        self.set_base_transformer(StandardScaler(**self.kwargs))

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'scaled(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class pca(Transformer):

    def __init__(self, model_config, project_settings):
        super(pca, self).__init__(model_config, project_settings )
        self.set_base_transformer(PCA(**self.kwargs))

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

    def fit_transform(self, X_touch, working_features,y):
        kwargs = dict()
        kwargs['y'] = y
        return super(loo_encoding, self).fit_transform(X_touch,working_features,**kwargs)

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        inclusion_patterns = self.inclusion_patterns
        assert len(inclusion_patterns) == 1
        pattern = inclusion_patterns[0]
        new_feature_name = 'loo(' + pattern  + ')'
        return [new_feature_name]


class LeaveOneOutEncoder:

    def __init__(self):
        pass

    def fit_transform(self,X_mat, working_features, y):
        loo_vals = list()
        for j in X_mat.columns:
            j_idx = X_mat[X_mat.loc[:,j] > 0].index
            try:
                yj = [y[ji] for ji in j_idx]
            except IndexError:
                pass
            cat_sum = sum(yj)
            cat_len = len(yj)
            for i in j_idx:
                y_loo_mean = (cat_sum-y[i]) / (cat_len - 1)
                loo_vals.append(y_loo_mean)
        assert len(loo_vals) == len(X_mat)
        return pd.DataFrame(pd.Series(loo_vals))



