import importlib

import pandas as pd

from manipulator import Manipulator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer

class TransformChain(Manipulator):

    def __init__(self,transformations, model_config, project_settings,original_columns=False):
        Manipulator.__init__(self, transformations, model_config, project_settings,original_columns)
        self.transformations = transformations

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
            working_features = self.working_features
            if working_features == None:
                inv_column_map = self.inv_column_map
                working_features = {v: k for k, v in inv_column_map.iteritems()}
            for d in transformations:
                transform_name = d.keys()[0]
                print "\t" + log_prefix + " Performing feature engineering (" + str(i) + '/' + str(
                    len(transformations)) + "): " + transform_name
                transform_class = getattr(engineering_module, transform_name)
                transform = transform_class(model_config)
                X_touch, X_untouched = transform.split(X_mat, working_features)
                X_touched = transform.fit(X_touch)
                X_transform, updated_col_map = transform.combine_and_reindex(X_touched, X_untouched, working_features)
                self._set_working_features(updated_col_map)
                if train == True:
                    self._output_features(transform_name)
                    self._update_working_data_feature_names_ref(transform_name)
                i += 1
        return X_transform


class Transform:

    def __init__(self,model_config):
        feature_eng_settings = model_config['feature_settings']['feature_engineering']
        self.base_transformer = None
        self.configure_transform(feature_eng_settings)

    def configure_transform(self,feature_eng_settings):
        transform_class = str(self.__class__).split('.')[-1:][0]
        transform_settings = self.fetch_transform_settings(feature_eng_settings, transform_class)
        if transform_settings.has_key('kwargs'):
            kwargs = transform_settings['kwargs']
        else:
            kwargs = dict()
        self.kwargs = kwargs
        self.inclusion_patterns = transform_settings['inclusion_patterns']

    def fetch_transform_settings(self,feature_eng_settings, transform_name):
        for item in feature_eng_settings:
            if item.keys()[0] == transform_name:
                transform_settings = item[transform_name]
            else:
                pass
        return transform_settings

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
        inv_working_features = {v: k for k, v in working_features.iteritems()}
        if inclusion_patterns == 'All':
            X_untouched = None
            X_touch = X_mat
            raise NotImplementedError
        else:
            for pattern in inclusion_patterns:
                pat_include_columns = filter(lambda x: pattern in x, col_names)
                include_columns = include_columns + pat_include_columns
            touch_indices = [int(inv_working_features[col_name]) for col_name in include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
            X_untouched = X_mat.loc[:,untouched_indices]
            X_touch = X_mat.loc[:,touch_indices]
        return X_touch, X_untouched

    def fit(self,X_touch):
        X_touched = self.base_transformer.fit_transform(X_touch)
        return X_touched

    def combine_and_reindex(self, X_touched, X_untouched, col_map):
        Xt_df = pd.DataFrame(X_touched)
        Xut_df = pd.DataFrame(X_untouched)
        if type(X_touched) == None:
            return X_untouched, col_map
        else:
            untouched_indices = Xut_df.columns.tolist()
            new_col_map = dict()
            ni = 0
            for oi in untouched_indices:
                col_name = col_map[oi]
                new_col_map[ni] = col_name
                ni += 1
            Xut_df.columns = range(len(untouched_indices))
            reindexed_col_idx = list()
            Xt_feat_names = self.gen_new_column_names(Xt_df,col_map)
            for feature in Xt_feat_names:
                new_col_map[ni] = feature
                reindexed_col_idx.append(ni)
                ni += 1
            Xt_df.columns = reindexed_col_idx
            assert ni == len(Xt_feat_names) + len(untouched_indices)
            X_transform = pd.merge(Xut_df, Xt_df, left_index=True, right_index=True)
            assert ni == X_transform.shape[1]
            assert len(X_transform) == len(Xut_df) == len(Xt_df)
            return X_transform, new_col_map

    def gen_new_column_names(self,Xt_df,col_map):
        pass

class basis_expansion(Transform):

    def __init__(self,model_config):
        Transform.__init__(self,model_config)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))

    def gen_new_column_names(self, Xt_df):
        Xt_feat_names = list()
        num_poly_features = Xt_df.shape[1]
        for i in range(num_poly_features):
            #base_feature_name = col_map[ind]
            #poly_feature_name = base_feature_name + '**' + str(i+1)
            poly_feature_name = 'polyfeature..' + str(i)  ##TODO: Make this name meaningful. Note this won't filter properly
            Xt_feat_names.append(poly_feature_name)       ## using exclusion_patterns
        assert len(Xt_feat_names) == Xt_df.shape[1]
        return Xt_feat_names


class normalize(Transform):

    def __init__(self,model_config):
        Transform.__init__(self,model_config)
        self.set_base_transformer(Normalizer(**self.kwargs))

    def gen_new_column_names(self,Xt_df,col_map):
        Xt_feat_names = list()
        for idx in Xt_df.columns.tolist():
            base_feature_name = col_map[idx]
            norm_feature_name = 'norm(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names
