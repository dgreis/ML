import importlib
import itertools
import copy
import os

import pandas as pd

from manipulator import Manipulator
from utils import flip_dict, find_project_dir, load_inv_column_map, load_clean_input_file_filepath
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from django.utils.text import slugify

class TransformChain(Manipulator):

    def __init__(self,starting_transformations, model_config, project_settings,original_columns=False):
        updated_transformations = list()
        for transformation in starting_transformations:
            engineering_module = importlib.import_module('feature.engineering')
            transformer_name = transformation.keys()[0].split('.')[-1:][0]
            transform_class = getattr(engineering_module, transformer_name)
            if transform_class.__bases__[0] == getattr(engineering_module,'TransformChain'):
                transform_chain_class = transform_class
                tc = transform_chain_class(starting_transformations,model_config,project_settings,original_columns)
                updated_transformations = updated_transformations + tc.transformations
            else:
                updated_transformations = updated_transformations + [transformation]
        expanded_model_config = copy.deepcopy(model_config)
        expanded_model_config['feature_settings']['feature_engineering'] = updated_transformations
        super(TransformChain,self).__init__(updated_transformations, expanded_model_config, project_settings,original_columns)
        self.transformations = updated_transformations

    def fetch_transform_settings(self,feature_eng_settings, transformer_name):
        for item in feature_eng_settings:
            if item.keys()[0] == transformer_name:
                transform_settings = item[transformer_name]
            else:
                pass
        return transform_settings

    def fit_transform(self, X_mat,train=True):
        if train == True:
            log_prefix = "[Train]"
        else:
            log_prefix = "[Test]"
        engineering_module = importlib.import_module('feature.engineering')
        transformations = self.transformations
        model_config = self.model_config
        i = 1
        if len(transformations) < 1:
            X_transform = X_mat
        else:
            for d in transformations:
                working_features = self.working_features
                if working_features is None:
                    inv_column_map = self.inv_column_map
                    working_features = flip_dict(inv_column_map)
                transformer_name = d.keys()[0]
                transformer_class_name = transformer_name.split('.')[-1:][0]
                print "\t" + log_prefix + " Performing feature engineering (" + str(i) + '/' + str(
                    len(transformations)) + "): " + transformer_name
                transform_class = getattr(engineering_module, transformer_class_name)
                feature_eng_settings = model_config['feature_settings']['feature_engineering']
                transformer_settings = self.fetch_transform_settings(feature_eng_settings, transformer_name)
                transformer_settings['model_name'] = model_config['model_name']
                transformer = transform_class(transformer_settings)
                X_touch, X_untouched = transformer.split(X_mat, working_features)
                X_touched, new_feat_names = transformer.fit_transform(X_touch, working_features)
                X_transform, updated_col_map = transformer.combine_and_reindex(X_touched, X_untouched
                                                                    ,working_features, new_feat_names)
                self._set_working_features(updated_col_map)
                if train == True:
                    self._output_features(transformer_name)
                    self._update_working_data_feature_names_ref(slugify(transformer_name))
                    if transformer.store:
                        artifact_dir = self.artifact_dir
                        transformer.store_output(X_transform,output_dir=artifact_dir)
                X_mat = X_transform
                i += 1
        return X_transform


class Transformer(object):

    def __init__(self, transformer_settings):
        super(Transformer,self).__init__()
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
        self.model_name = transformer_settings['model_name']
        self.inclusion_patterns = transformer_settings['inclusion_patterns']

    def set_base_transformer(self,transformer_instance):
        self.base_transformer = transformer_instance

    def set_inclusion_patterns(self,inclusion_patterns):
        self.inclusion_patterns = inclusion_patterns

    def split(self, X_mat, working_features):
        assert getattr(self, 'inclusion_patterns') != None
        include_columns = list()
        col_indices = working_features.keys()
        col_names = working_features.values()
        inclusion_patterns = self.inclusion_patterns
        inv_working_features = flip_dict(working_features)
        if inclusion_patterns == ['All']:
            X_untouched = pd.DataFrame()
            X_touch = X_mat
        else:
            for pattern in inclusion_patterns:
                len_pat = len(pattern)
                pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
                include_columns = include_columns + pattern_begin_cols
            non_inter_include_columns = filter(lambda x: 'x%x' not in x, include_columns)
            touch_indices = [int(inv_working_features[col_name]) for col_name in non_inter_include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
            X_untouched = X_mat.loc[:,untouched_indices]
            X_touch = X_mat.loc[:,touch_indices]
        return X_touch, X_untouched

    def fit_transform(self, X_touch, working_features):
        X_touched = self.base_transformer.fit_transform(X_touch)
        orig_tcol_idx = X_touch.columns
        Xt_feat_names = self.gen_new_column_names(orig_tcol_idx, working_features)
        if type(X_touched) != pd.core.frame.DataFrame: #TODO: fix this when I move out of pandas
            rdf = pd.DataFrame(X_touched)
        else:
            rdf = X_touched
        assert len(Xt_feat_names) == X_touched.shape[1]
        return rdf, Xt_feat_names

    def store_output(self,X_mat,output_dir): #TODO: When I need to output filter, make this an abstract method in new Manipulator class
        model_name = self.model_name
        transform_name = str(self.__class__).split('.')[-1:][0]
        output_file_name = slugify(model_name + '-' + transform_name)
        X_mat.to_csv(output_dir + '/' + output_file_name + '.txt', header=False, index=False, sep='\t')  # TODO: fix this when I move out of pandas

    def combine_and_reindex(self, Xt_df, Xut_df, working_features, Xt_feat_names):
        untouched_indices = Xut_df.columns.tolist()
        new_col_map = dict()
        ni = 0
        for oi in untouched_indices:
            col_name = working_features[oi]
            new_col_map[ni] = col_name
            ni += 1
        Xut_df.columns = range(len(untouched_indices))
        reindexed_col_idx = list()
        for feature in Xt_feat_names:
            new_col_map[ni] = feature
            reindexed_col_idx.append(ni)
            ni += 1
        Xt_df.columns = reindexed_col_idx
        assert ni == len(Xt_feat_names) + len(untouched_indices)
        if len(Xut_df) > 0:
            X_transform = pd.merge(Xut_df, Xt_df, left_index=True, right_index=True)
            assert len(X_transform) == len(Xut_df) == len(Xt_df)
        else:
            X_transform = Xt_df
        assert ni == X_transform.shape[1]
        return X_transform, new_col_map

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        raise NotImplementedError

class basis_expansion(Transformer):

    def __init__(self, transformer_settings):
        super(basis_expansion, self).__init__(transformer_settings)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))

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
        raw_interaction_strs = filter(lambda x: x.keys()[0] == 'interaction_terms',  transformations)[0]['interaction_terms']['interactions']
        compact_interactions = [eval(ris) for ris in raw_interaction_strs]
        expanded_transformations = list()
        transformation_names = [d.keys()[0] for d in transformations]
        inter_tranform_order = transformation_names.index('interaction_terms')
        if inter_tranform_order == 0:
            feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            prior_transform = transformation_names[inter_tranform_order-1]
            artifact_dir = find_project_dir(project_settings) + '/' + 'model_artifacts'
            model_artifact_filepaths = os.listdir(artifact_dir)
            candidate_files = filter(lambda x: prior_transform in x, model_artifact_filepaths)
            assert len(candidate_files) == 1
            prior_transform_feature_names_rel_filepath = candidate_files[0]
            feature_names_filepath = artifact_dir + '/' + prior_transform_feature_names_rel_filepath
        inv_column_map = load_inv_column_map(feature_names_filepath)
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
        super(interaction_terms, self).__init__(expanded_transformations, model_config, project_settings,original_columns)

class normalize(Transformer):

    def __init__(self, transform_settings):
        super(normalize,self).__init__(transform_settings)
        self.set_base_transformer(Normalizer(**self.kwargs))

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'norm(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names


class standard_scale(Transformer):

    def __init__(self, transform_settings):
        super(standard_scale,self).__init__(transform_settings)
        self.set_base_transformer(StandardScaler(**self.kwargs))

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'scaled(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class pca(Transformer):

    def __init__(self, transform_settings):
        super(pca,self).__init__(transform_settings)
        self.set_base_transformer(PCA(**self.kwargs))

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        num_comps = self.base_transformer.n_components
        new_col_list = list()
        for i in range(num_comps):
            new_col_name = 'pc_' + str(i)
            new_col_list.append(new_col_name)
        return new_col_list
