import pandas as pd
import numpy as np

from feature.engineering import TransformChain
from feature.selection import FilterChain
from utils import find_data_dir, load_clean_input_file_filepath, load_inv_column_map, flip_dict

class Manager:

    def __init__(self,model_config, project_settings):
        transformations = model_config['feature_settings']['feature_engineering']
        filters = model_config['feature_settings']['feature_selection']

        transformer_chain = TransformChain(transformations, model_config, project_settings)
        filter_chain = FilterChain(filters, model_config, project_settings)
        self.model_config = model_config
        self.project_settings = project_settings

        initialized_manipulations =  filter_chain.filters + transformer_chain.transformations
        leak_enforcer = LeakEnforcer(initialized_manipulations)
        self.leak_enforcer = leak_enforcer
        self.return_train_val = leak_enforcer.has_peekers

        for chain in [filter_chain, transformer_chain]:
            chain.set_leak_enforcer(leak_enforcer)
        self.filter_chain = filter_chain
        self.transformer_chain = transformer_chain

    def return_fold_dev_val_ind(self,fold_num):
        model_config = self.model_config
        folds_map = model_config['folds_map']
        ind_dev, ind_val = folds_map[fold_num]
        if self.return_train_val:
            return np.append(ind_dev,ind_val), ind_val
        else:
            return ind_dev, ind_val

    def fit_transform(self, X, y, dataset_name):

        model_config = self.model_config

        fc = self.filter_chain
        tc = self.transformer_chain

        if model_config['feature_settings']['select_before_eng']:
            X_1st, y_1st = fc.fit_transform(X, y, dataset_name)
            X_2nd, y_2nd = tc.fit_transform(X_1st, y_1st,dataset_name)
        else:
            X_1st, y_1st = tc.fit_transform(X, y,dataset_name)
            X_2nd, y_2nd = fc.fit_transform(X_1st, y_1st, dataset_name)

        assert X.shape[0] == len(y)

        return X_2nd, y_2nd


    def transform(self,X, y, dataset_name):

        model_config = self.model_config

        fc = self.filter_chain
        tc = self.transformer_chain

        if model_config['feature_settings']['select_before_eng']:
            X_1st, y_1st = fc.transform(X,y,dataset_name)
            X_2nd, y_2nd = tc.transform(X_1st,y_1st, dataset_name)
        else:
            X_1st, y_1st = tc.transform(X,y, dataset_name)
            X_2nd, y_2nd = fc.transform(X_1st,y_1st, dataset_name)

        return X_2nd, y_2nd

    def load_clean_datasets(self,dataset_name, project_settings):

        assert dataset_name in ['train_val','train','val','test']

        clean_input_files = project_settings['clean_input_files']

        X_name, y_name = [('X_' + d, 'y_' + d) for d in [dataset_name]][0]

        data_dir = find_data_dir(project_settings)
        data = dict()

        if project_settings.has_key('take_nth_row'):
            take_nth_row = project_settings['take_nth_row']
        else:
            take_nth_row = 1
        X_abs_filepath = data_dir + '/' + clean_input_files[X_name]
        X = pd.read_csv(X_abs_filepath, sep="\s+", engine='python', header=None, skiprows=lambda i: i % take_nth_row != 0)


        y_mat_file_path = data_dir + '/' + clean_input_files[y_name]
        y_mat = pd.read_csv(y_mat_file_path, sep="\s+", engine='python', header=None, skiprows= lambda i: i % take_nth_row != 0)
        y = y_mat.iloc[:, 0].tolist()

        data[dataset_name] = (X,y)

        return data

class LeakEnforcer:

    def __init__(self, manipulations_list):
        manipulator_map = dict()
        has_peekers = False
        for i in range(len(manipulations_list)):
            manipulator_entry = manipulations_list[i]
            manipulator_name = manipulator_entry.keys()[0]
            initialized_manipulator = manipulator_entry[manipulator_name]['initialized_manipulator']
            manipulator_peeking_status = initialized_manipulator.validation_peeking
            manipulator_map[manipulator_name] = { 'initialized_manipulator' : initialized_manipulator,
                                                  'manipulator_peeking_status' : manipulator_peeking_status
                                                }
        if True in [item['manipulator_peeking_status'] for item in manipulator_map.values()]:
            has_peekers = True
        self.manipulator_map = manipulator_map
        self.has_peekers = has_peekers

    def check_for_leak(self,X_mat):
        manipulator_map = self.manipulator_map
        initialized_manipulators = [item['initialized_manipulator'] for item in manipulator_map.values()]
        self.check_consistent_folds(initialized_manipulators)
        manipulator_i = initialized_manipulators[0]
        folds_map = manipulator_i.model_config['folds_map']
        fold_i =  manipulator_i.model_config['fold_i']
        leaking_candidates = folds_map[fold_i][1]
        if len(set(leaking_candidates).intersection(X_mat.index.values)) > 0:
            return True
        else:
            return False

    def check_leak_allowed(self,manipulator_name):
        manipulator_map = self.manipulator_map
        return manipulator_map[manipulator_name]['manipulator_peeking_status']

    def check_consistent_folds(self, initialized_manipulators):
        manipulator_folds_map = dict()
        for i in range(len(initialized_manipulators)):
            initialized_manipulator = initialized_manipulators[0]
            assert initialized_manipulator.model_config.has_key('folds_map')
            manipulator_folds_map[i] = initialized_manipulator.model_config['folds_map']
        assert all(value == manipulator_folds_map[0] for value in manipulator_folds_map.values())
        #Above confirms that all manipulators have same CV fold information

    def remove_leaking_indices(self,X,y):
        manipulator_map = self.manipulator_map
        initialized_manipulators = [item['initialized_manipulator'] for item in manipulator_map.values()]
        manipulator_i = initialized_manipulators[0]
        folds_map = manipulator_i.model_config['folds_map']
        fold_i = manipulator_i.model_config['fold_i']
        leaking_indices = folds_map[fold_i][1]
        X_noleak = X.drop(leaking_indices)
        y_noleak = list(pd.Series(y,index=X.index).drop(leaking_indices))
        return X_noleak, y_noleak
