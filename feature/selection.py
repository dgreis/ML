import pandas as pd
import importlib

from manipulator import ManipulatorChain, Manipulator
from utils import flip_dict, load_inv_column_map, load_clean_input_file_filepath
from algorithms.classification import Decision_Tree_Classifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV

class FilterChain(ManipulatorChain):

    def __init__(self,filters, model_config, project_settings,original_columns=False):
        model_config['feature_settings']['order'] = 0
        ManipulatorChain.__init__(self, filters, model_config, project_settings, original_columns)
        self.filters = filters

    def fit(self,X_mat,y):
        """To be fit method of filter_chain class"""
        selection_module = importlib.import_module('feature.selection')
        filters = self.filters
        model_config = self.model_config
        model_config['feature_settings']['order'] = -1
        project_settings = self.project_settings
        #self._pass_y_to_self(y)
        X_filt = X_mat
        i = 1
        if len(filters) < 1:
            pass
        else:
            for d in filters:
                filter_name = d.keys()[0]
                print "\t[Train] Performing model selection (" + str(i) + '/' + str(len(filters)) + "): " + filter_name
                filter_class = getattr(selection_module,filter_name)
                fit_args = self._get_args(filter_class, 'fit')
                additional_args = filter(lambda x: x not in ['X_mat','y'], fit_args)
                model_config['feature_settings']['order'] += 1
                filter_instance = filter_class(model_config,project_settings)
                kwargs = dict()
                for arg in additional_args:
                    kwargs[arg] = getattr(self,arg)
                filter_instance.fit(X_mat,y,**kwargs)
                features = filter_instance.features
                reindexed_features = filter_instance.reindex(features)
                filter_instance.features = reindexed_features
                filter_instance.output_features()
                d[filter_name]['initialized_filter'] = filter_instance
                i += 1

    def transform(self, X_mat,y,dataset_name=None):
        filters = self.filters
        X_filt, y_filt = X_mat, y
        if dataset_name is not None:
            log_prefix = dataset_name
        else:
            log_prefix = "Non-training"
        num_filters = len(filters)
        if num_filters > 0:
            print "\t["+log_prefix+"] Filtering selected features"
            for i in range(num_filters):
                filter_name = filters[i].keys()[0]
                filter_instance = filters[i][filter_name]['initialized_filter']
                _, X_filt, _, y_filt = filter_instance.split(X_filt, y, )
                X_filt.columns = filter_instance.features.keys()
        return X_filt, y_filt

    def det_prior_feature_names_filepath(self,model_config):
        pass

class Filter(Manipulator):

    def __init__(self, model_config, project_settings):
        manipulations = model_config['feature_settings']['feature_selection']
        super(Filter,self).__init__(model_config, project_settings,manipulations)

    def fetch_filter_settings(self,filter_name):
        model_config = self.model_config
        feature_selection_settings = model_config['feature_settings']['feature_selection']
        for item in feature_selection_settings:
            if item.keys()[0] == filter_name:
                filter_settings = item[filter_name]
            else:
                pass
        return filter_settings

    def _store_indices_and_features(self,untouched_indices,touch_indices):
        self.untouched_indices = untouched_indices
        self.touch_indices = touch_indices
        feature_names_filepath = self.prior_manipulator_feature_names_filepath
        inv_column_map = load_inv_column_map(feature_names_filepath)
        column_map = flip_dict(inv_column_map)
        filtered_features = {k: v for k, v in column_map.iteritems() if k in untouched_indices}
        self.features = filtered_features

    def det_prior_feature_names_filepath(self,model_config):
        project_settings = self.project_settings
        if model_config['feature_settings']['select_before_eng']:
            prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            transformations = model_config['feature_settings']['feature_engineering']
            num_transformations = len(transformations)
            if num_transformations > 0:
                last_transformer_name = transformations[-1:][0].keys()[0]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(last_transformer_name)
            else:
                prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings,'feature_names')
        return prior_manipulator_feature_names_filepath

class exclude_features(Filter):
    """
    Example yaml usage:

    Models:
      <model name>:
        feature_settings:
          feature_selection:
            - exclude_features:
                patterns:
                  - <matching pattern>
    """
    def __init__(self, model_config, project_settings):
        super(exclude_features, self).__init__(model_config, project_settings)
        self.patterns = self.fetch_filter_settings('exclude_features')

    def fit(self,**kwargs):
        prior_manipulator_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        inv_column_map = load_inv_column_map(prior_manipulator_feature_names_filepath)
        prior_features = flip_dict(inv_column_map)
        col_names = prior_features.values()
        col_indices = prior_features.keys()
        exclude_columns = list()
        exclusion_patterns = self.patterns['patterns']
        for pattern in exclusion_patterns:
            len_pat = len(pattern)
            pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
            exclude_columns = exclude_columns + pattern_begin_cols
        touch_indices = [int(inv_column_map[col_name]) for col_name in exclude_columns]
        untouched_indices = list(set(col_indices).difference(set(touch_indices)))
        self._store_indices_and_features(untouched_indices,touch_indices)

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
    """
    Example yaml usage:

    Models:
      <Model Name>
        feature_settings:
          feature_selection:
           - tree_based:
              keyword_arg_settings: {}
              other_options: {}
    """
    def __init__(self,model_config,project_settings):
        super(tree_based,self).__init__(model_config,project_settings)
        self.tree_based_settings = self.fetch_filter_settings('tree_based')

    def fit(self,X_mat,y,project_settings):

        tree_based_settings = self.tree_based_settings
        tree_based_settings['model_name'] = 'tree-based-feature-selection' #TODO: add this onto existing model name
        clf = Decision_Tree_Classifier(tree_based_settings,project_settings)
        clf = clf.fit(X_mat, y)
        if clf.gen_output_flag:
            clf.gen_output()

        if tree_based_settings['other_options'].has_key('selection_threshold'):
            selection_threshold = tree_based_settings['other_options']['selection_threshold']
        else:
            selection_threshold = None
        model = SelectFromModel(clf, prefit=True, threshold=selection_threshold)
        X_filt = pd.DataFrame(model.transform(X_mat))
        untouched_indices = X_filt.columns.tolist()
        touch_indices = list(set(X_mat.columns).difference(set(untouched_indices)))
        self._store_indices_and_features(untouched_indices,touch_indices)

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
