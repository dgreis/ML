import pandas as pd
import importlib

from manipulator import Manipulator
from utils import flip_dict
from algorithms.classification import Decision_Tree_Classifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV

class FilterChain(Manipulator):

    def __init__(self,filters, model_config, project_settings,original_columns=False):
        Manipulator.__init__(self, filters, model_config, project_settings,original_columns)
        self.filters = filters

    def fit_transform(self,X_train,y_train):
        """To be fit method of filter_chain class"""
        selection_module = importlib.import_module('feature.selection')
        filters = self.filters
        model_config = self.model_config
        self._pass_y_to_self(y_train)
        X_filt = X_train
        orig_inv_col_map = self.inv_column_map
        orig_col_map = flip_dict(orig_inv_col_map)
        init_num_feats = len(X_train.columns)  #this might need to be changed when no longer pandas
        i = 1
        if len(filters) < 1:
            pass
        else:
            for d in filters:
                filter_name = d.keys()[0]
                print "\t[Train] Performing model selection (" + str(i) + '/' + str(len(filters)) + "): " + filter_name
                filter_class = getattr(selection_module,filter_name)
                apply_args = self._get_args(filter_class, 'apply')
                additional_args = filter(lambda x: x not in ['X_mat'], apply_args)
                filter_instance = filter_class(model_config)
                kwargs = dict()
                for arg in additional_args:
                    kwargs[arg] = getattr(self,arg)
                X_filt = filter_instance.apply(X_filt,**kwargs)
                i += 1
            working_features = self.working_features
            if working_features == None:
                working_features = orig_col_map
            ni = 0
            updated_col_map = dict()
            for oi in X_filt.columns.tolist():
                updated_col_map[ni] = working_features[oi]
                ni += 1
            self._set_working_features(updated_col_map)
            filt_num_feats = len(self.working_features)
            if filt_num_feats < init_num_feats:
                print "\t\tAfter model selection, number of features now " + str(filt_num_feats) +", down from " + str(init_num_feats) +". " + \
                    "See project model_artifacts folder for more info."
            else:
                print "\t\tNo features elminated in model selection process"
            self._output_features('selected-features')
            self._update_working_data_feature_names_ref('selected-features')
        X_filt.columns = range(X_filt.shape[1])
        return X_filt

    def transform(self, X_mat,original_columns=False,dataset_name=None):
        filters = self.filters
        if dataset_name is not None:
            log_prefix = dataset_name
        else:
            log_prefix = "Non-training"
        if len(filters) > 0:
            print "\t["+log_prefix+"] Filtering selected features"
            working_features = self.working_features  # { ind : feat_name }
            #if not original_columns:
            #    X_filt = X_mat.iloc[:, working_features.keys()]
            #else:
            filt_indices = list()
            inv_working_features = flip_dict(working_features)
            orig_inv_col_map = self.inv_column_map
            for col in inv_working_features:
                if orig_inv_col_map.has_key(col):
                    filt_indices.append(inv_working_features[col])
                else:
                    pass
            X_filt = X_mat.iloc[:,filt_indices]
        else:
            X_filt = X_mat
        return X_filt

class Filter:

    def __init__(self, model_config):
        self.model_config = model_config

    def fetch_filter_settings(self,filter_name):
        model_config = self.model_config
        feature_selection_settings = model_config['feature_settings']['feature_selection']
        for item in feature_selection_settings:
            if item.keys()[0] == filter_name:
                filter_settings = item[filter_name]
            else:
                pass
        return filter_settings

class exclusion_patterns(Filter):

    def __init__(self,model_config):
        Filter.__init__(self,model_config)
        self.exclusion_patterns = self.fetch_filter_settings('exclusion_patterns')

    def apply(self, X_mat, inv_column_map):
        exclude_columns = list()
        col_names = inv_column_map.keys()
        exclusion_patterns = self.exclusion_patterns
        for pattern in exclusion_patterns:
            len_pat = len(pattern)
            pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
            exclude_columns = exclude_columns + pattern_begin_cols
        exclude_indices = [int(inv_column_map[col_name]) for col_name in exclude_columns]
        X_mat_filt = X_mat.drop(axis=1,labels=exclude_indices)
        return X_mat_filt

class inclusion_patterns(Filter):

    def __init__(self,model_config):
        super(inclusion_patterns,self).__init__(model_config)
        self.inclusion_patterns = self.fetch_filter_settings('inclusion_patterns')

    def apply(self, X_mat, inv_column_map):
        include_columns = list()
        col_names = inv_column_map.keys()
        inclusion_patterns = self.inclusion_patterns
        for pattern in inclusion_patterns:
            len_pat = len(pattern)
            pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
            include_columns = include_columns + pattern_begin_cols
        include_indices = [int(inv_column_map[col_name]) for col_name in include_columns]
        X_mat_filt = X_mat.loc[:,include_indices]
        return X_mat_filt

class l1_based(Filter):

    def __init__(self,model_config):
        Filter.__init__(self,model_config)
        self.l1_settings = self.fetch_filter_settings('l1_based')

    def apply(self,X_mat,y_train):
        l1_settings = self.l1_settings
        method = l1_settings['method']
        kwargs = l1_settings['kwargs']
        kwargs['penalty'] = 'l1'
        if method == "LinearSVC":
            l1_model = LinearSVC(**kwargs)
        elif method == "LogisticRegression":
            l1_model = LogisticRegression(**kwargs)
        else:
            raise NotImplementedError
        l1_model.fit(X_mat,y_train)
        s = pd.Series(l1_model.coef_.sum(0))
        nonzero_features = s[abs(s) > 0.0001].index.tolist()
        X_filt = X_mat.iloc[:,nonzero_features]
        return X_filt

class tree_based(Filter):

    def __init__(self,model_config):
        Filter.__init__(self,model_config)
        self.tree_based_settings = self.fetch_filter_settings('tree_based') #TODO: Make more like Transform constructor

    def apply(self,X_mat,y_train,project_settings):

        tree_based_settings = self.tree_based_settings
        tree_based_settings['model_name'] = 'tree-based-feature-selection' #TODO: add this onto existing model name
        clf = Decision_Tree_Classifier(tree_based_settings,project_settings)
        clf = clf.fit(X_mat, y_train)
        if clf.gen_output_flag:
            clf.gen_output()

        if tree_based_settings['other_options'].has_key('selection_threshold'):
            selection_threshold = tree_based_settings['other_options']['selection_threshold']
        else:
            selection_threshold = None
        model = SelectFromModel(clf, prefit=True, threshold=selection_threshold)
        X_filt = pd.DataFrame(model.transform(X_mat))
        return X_filt

class recursive_feature_elimination(Filter):

    def __init__(self,model_config):
        Filter.__init__(self,model_config)
        self.rfe_settings = self.fetch_filter_settings('recursive_feature_elimination')


    def apply(self,X_mat,y_train):
        rfe_settings = self.rfe_settings
        kwargs = rfe_settings['kwargs']
        #estimator_class_name = kwargs['estimator']
        #current_mod = importlib.import_module('feature.selection')
        #estimator_class = getattr(current_mod,estimator_class_name)
        #estimator = estimator_class()
        estimator = SVC(kernel='linear')
        remaining_kwargs_keys = filter(lambda x: x not in ['estimator'],kwargs.keys())
        remaining_kwargs = {k: kwargs[k] for k in remaining_kwargs_keys}
        selector = RFECV(estimator,**remaining_kwargs)
        X_filt = selector.fit_transform(X_mat,y_train)
        return pd.DataFrame(X_filt)
