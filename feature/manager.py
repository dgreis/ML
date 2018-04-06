import pandas as pd
import inspect
import importlib
import os

from utils import find_data_dir


class FeatureManager:

    def __init__(self, model_config, project_settings):
        """Default order is to engineer before selection"""
        if model_config['feature_settings']['feature_engineering']['post_selection'] == True:
            self.select_first = True
        else:
            self.select_first = False
        self.transformations = model_config['feature_settings']['feature_engineering'].keys()
        self.filters = model_config['feature_settings']['feature_selection']

        self.model_config = model_config

        data_dir = find_data_dir(project_settings)

        column_names_filepath = data_dir + '/' + project_settings['final_files']['feature_names']
        self.inv_column_map = self.load_column_map(column_names_filepath)

    def load_column_map(self, filepath):
        df = pd.read_csv(filepath, sep="\s+", engine='python', names=['col_name'])
        return pd.Series(df.index, index=df.col_name).to_dict()

    def select_features(self,X_train,y_train):
        selection_module = importlib.import_module('feature.selection')
        filters = self.filters
        model_config = self.model_config
        X_filt = X_train
        for d in filters:
            filter_name = d.keys()[0]
            filter_class = getattr(selection_module,filter_name)
            additional_args = self._get_args(filter_class)
            filter = filter_class(model_config)
            kwargs = dict()
            for arg in additional_args:
                kwargs[arg] = getattr(self,arg)
            X_filt = filter.apply(X_filt,**kwargs)
        self._set_selected_features(X_filt)
        return X_filt

    def _set_selected_features(self,X_mat):
        self.selected_features = X_mat.columns.tolist()

    def _get_args(self,filter_class):
        args = getattr(inspect.getargspec(filter_class.apply),'args')
        additional_args = args[2:]
        return additional_args

    def engineer_features(self,X_mat):
        print "TODO: implement this"
        return X_mat

    def filter_selected_features(self,X_mat):
        assert hasattr(self,'selected_features')
        selected_features = self.selected_features
        X_filt = X_mat.iloc[:,selected_features]
        return X_filt