import pandas as pd
import importlib
import copy

from manipulator import ManipulatorChain, Manipulator
from utils import flip_dict, load_inv_column_map, load_clean_input_file_filepath
from algorithms.classification import DecisionTreeClassifier
from algorithms.regression import DecisionTreeRegressor

from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression

class FilterChain(ManipulatorChain):

    def __init__(self,filters, model_config, project_settings):
        ManipulatorChain.__init__(self, filters, model_config, project_settings)
        self.filters = filters

    def fit_transform(self,X_mat,y,dataset_name):
        """To be fit method of filter_chain class"""
        selection_module = importlib.import_module('feature.selection')
        filters = self.filters
        model_config = self.model_config
        model_config['feature_settings']['order'] = 0
        project_settings = self.project_settings
        X_filt, y_filt = X_mat, y
        i = 1
        if len(filters) < 1:
            pass
        else:
            for d in filters:
                filter_name = d.keys()[0]
                #print "\t[" + dataset_name + "] Performing model selection (" + str(i) + '/' + str(len(filters)) + "): " + filter_name
                filter_class = getattr(selection_module,filter_name)
                fit_args = self._get_args(filter_class, 'fit')
                additional_args = filter(lambda x: x not in ['X_mat','y'], fit_args)
                filter_instance = filter_class(model_config,project_settings)
                model_config['feature_settings']['order'] += 1
                kwargs = dict()
                for arg in additional_args:
                    kwargs[arg] = getattr(self,arg)
                filter_instance.fit(X_filt,y_filt,**kwargs)
                features = filter_instance.features
                reindexed_features = filter_instance.reindex(features)
                #filter_instance.features = reindexed_features
                setattr(filter_instance,'features',reindexed_features)
                filter_instance.output_features()
                d[filter_name]['initialized_filter'] = filter_instance
                _, X_filt, y_filt, _ = filter_instance.split(X_filt, y)
                assert True not in pd.isnull(X_filt).any(1).value_counts() #TODO: pandas dependent
                X_filt.columns = filter_instance.features.keys()
                i += 1
        return X_filt, y_filt

    def transform(self, X_mat,y,dataset_name):
        #TODO: select according to final filter
        filters = self.filters
        log_prefix = dataset_name
        num_filters = len(filters)
        X_filt = X_mat
        if num_filters > 0:
            #print "\t["+log_prefix+"] Filtering selected features"
            first_filter = self._fetch_initialized_filter(filters[0])
            final_filter = self._fetch_initialized_filter(filters[-1:][0])
            prior_features_filepath = first_filter.prior_manipulator_feature_names_filepath
            orig_inv_column_map = load_inv_column_map(prior_features_filepath)
            final_filter_features = final_filter.features
            inv_column_map = flip_dict(final_filter_features)
            filtered_indices = list()
            for col in inv_column_map:
                filtered_indices.append(orig_inv_column_map[col])
            sorted_filtered_indices = sorted(filtered_indices)
            X_filt = X_mat.loc[:,sorted_filtered_indices]
            X_filt.columns = final_filter.features.keys()
        assert True not in pd.isnull(X_filt).any(1).value_counts()  # TODO: pandas dependent
        return X_filt, y

    def _fetch_initialized_filter(self,entry):
        assert len(entry.keys()) == 1
        key = entry.keys()[0]
        return entry[key]['initialized_filter']

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

    def det_prior_init_feature_names_filepath(self, model_config):
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


class l1_based(Filter):
    """
    Sample yaml usage:
    <model name>:
        base_algorithm: <algorithm>
        feature_settings:
          feature_selection:
           - l1_based:
               method: "LinearSVC"
               kwargs:
                 dual: False
    """
    def __init__(self,model_config,project_settings):
        Filter.__init__(self,model_config,project_settings)
        self.l1_settings = self.fetch_filter_settings('l1_based')

    def fit(self, X_mat, y):
        l1_settings = self.l1_settings
        method = l1_settings['method']
        kwargs = l1_settings['kwargs']
        if method == "LinearSVC":
            kwargs['penalty'] = 'l1'
            l1_model = LinearSVC(**kwargs)
        elif method == "LogisticRegression":
            kwargs['penalty'] = 'l1'
            l1_model = LogisticRegression(**kwargs)
        elif method == "Lasso":
            l1_model = Lasso(**kwargs)
        else:
            raise NotImplementedError
        l1_model.fit(X_mat, y)
        if method == "LinearSVC":
            s = pd.Series(l1_model.coef_.sum(0))
        elif method == "Lasso":
            s = pd.Series(l1_model.coef_)
        else:
            raise NotImplementedError
        untouched_indices = s[abs(s) > 0.0001].index.tolist()
        touch_indices = list(set(X_mat.columns).difference(set(untouched_indices)))
        self._store_indices_and_features(untouched_indices,touch_indices)

class tree_based(Filter):
    """
    Example yaml usage:

    Models:
      <Model Name>
        feature_settings:
          feature_selection:
           - tree_based:
              method: <str> "classification" or "regression"
              keyword_arg_settings: {}
              other_options: {}
    """
    def __init__(self,model_config,project_settings):
        super(tree_based,self).__init__(model_config,project_settings)
        tree_model_config = self.fetch_filter_settings('tree_based')
        tree_model_config['feature_settings'] = model_config['feature_settings']
        tree_model_config['model_name'] = model_config['model_name'] + '-tree-based-feature-selection'
        method = tree_model_config['method']
        if method == 'classification':
            model = DecisionTreeClassifier(tree_model_config, project_settings, mode='filter')
            #TODO: Consider parameterizing this to use diff algo, i.e. Regression?
        elif method == 'regression':
            model = DecisionTreeRegressor(tree_model_config,project_settings,mode='filter')
        else:
            raise NotImplementedError
        setattr(self,'model',model)


    def fit(self,X_mat,y):
        tree_model_config = self.fetch_filter_settings('tree_based')
        model = self.model
        model.fit(X_mat, y)
        if model.gen_output_flag:
            model.gen_output()
        if tree_model_config['other_options'].has_key('selection_threshold'):
            selection_threshold = tree_model_config['other_options']['selection_threshold']
        else:
            selection_threshold = None
        selector = SelectFromModel(model, prefit=True, threshold=selection_threshold)
        ir = pd.Series(selector.get_support())
        untouched_indices = ir[ir == True].index
        touch_indices = list(set(X_mat.columns).difference(set(untouched_indices)))
        self._store_indices_and_features(untouched_indices,touch_indices)


class f_based(Filter):
    """
     Example yaml usage:

     Models:
       <Model Name>
         feature_settings:
           feature_selection:
            - f_based:
               method: <str> i.e. 'regression or classif'
               keyword_arg_settings: {}
               other_options:
                    k: <int>
     """

    def __init__(self,model_config,project_settings):
        super(f_based,self).__init__(model_config,project_settings)
        self.f_based_settings = self.fetch_filter_settings('f_based')

    def fit(self,X_mat,y):
        f_based_settings = self.f_based_settings
        method = f_based_settings['method']
        if method == "classif":
            f_model = f_classif
        elif method == 'regression':
            f_model = f_regression
        else:
            raise NotImplementedError
        other_options = f_based_settings['other_options']
        k = other_options['k']

        selector = SelectKBest(f_model, k=k).fit(X_mat, y)
        ir = pd.Series(selector.get_support())
        untouched_indices = ir[ir == True].index
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
