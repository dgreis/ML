import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


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
            pat_exclude_columns = filter(lambda x: pattern in x, col_names)
            exclude_columns = exclude_columns + pat_exclude_columns
        exclude_indices = [int(inv_column_map[col_name]) for col_name in exclude_columns]
        X_mat_filt = X_mat.drop(axis=1,labels=exclude_indices)
        return X_mat_filt

class l1_based(Filter):

    def __init__(self,model_config):
        Filter.__init__(self,model_config)
        self.l1_settings = self.fetch_filter_settings('l1_based')

    def apply(self,X_train,y_train):
        l1_settings = self.l1_settings
        method = l1_settings['method']
        kwargs = l1_settings['kwargs']
        if method == "LinearSVC":
            kwargs['penalty'] = 'l1'
            l1_model = LinearSVC(**kwargs)
        else:
            raise NotImplementedError
        l1_model.fit(X_train,y_train)
        s = pd.Series(l1_model.coef_.sum(0))
        nonzero_features = s[s>0].index.tolist()
        #model = SelectFromModel(l1_model,prefit=True) #This is not the same as the way I'm implementing it!
        #X_filt = pd.DataFrame(model.transform(X_train))
        X_filt = X_train.iloc[:,nonzero_features]
        return X_filt

