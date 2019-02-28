from __future__ import print_function

import importlib

import pandas as pd
import numpy as np

from collections import OrderedDict

from feature.engineering import TransformChain, Cleaner, delete_obs
from feature.selection import FilterChain
from utils import find_data_dir, load_clean_input_file_filepath, load_inv_column_map, flip_dict

class Manager:

    def __init__(self,model_config, project_settings):

        manipulations = model_config['feature_settings']['manipulations']

        selection_module = importlib.import_module('feature.selection')
        engineering_module = importlib.import_module('feature.engineering')
        manipulator_map = dict()
        chain_plan = list()
        i = 0
        for m in manipulations:
            manipulator_id = m.keys()[0]
            manipulator_class_name = manipulator_id.split('.')[-1:][0]
            if hasattr(engineering_module, manipulator_class_name):
                manipulator_type = 'transformer'
            elif hasattr(selection_module, manipulator_class_name):
                manipulator_type = 'filter'
            else:
                raise Exception
            manipulator_map[i] = { 'manipulator_name' : manipulator_id, 'manipulator_type' : manipulator_type }
            if i == 0:
                chain_plan.append((manipulator_type, 0))
            elif chain_plan[-1][0] != manipulator_type:
                chain_plan.append((manipulator_type, i))
            i += 1

        len_chain = len(chain_plan)
        counter = 1
        chain_of_chains = list()
        for item in chain_plan:
            if counter < len_chain:
                end_idx = chain_plan[counter][1]
            else:
                end_idx = len(manipulations)
            beg_idx = item[1]
            chain_manipulations = manipulations[beg_idx:end_idx]
            if item[0] == 'transformer':
                chain = TransformChain('_' + str(counter - 1) + '_tc_', chain_manipulations, model_config, project_settings)
            else:
                chain = FilterChain('_' + str(counter - 1) + '_fc_', chain_manipulations, model_config, project_settings)
            chain_of_chains.append(chain)
            counter += 1

        self.model_config = model_config
        self.project_settings = project_settings

        initialized_manipulators = list()
        for chain in chain_of_chains:
            initialized_manipulators = initialized_manipulators + chain.manipulations
        leak_enforcer = LeakEnforcer(initialized_manipulators, model_config)
        self.leak_enforcer = leak_enforcer

        for chain in chain_of_chains:
            chain.set_leak_enforcer(leak_enforcer) #TODO: Any reason for this to be separate loop?

        self.chain_of_chains = chain_of_chains

    def return_fold_dev_val_ind(self,fold_num):
        model_config = self.model_config
        folds_map = model_config['folds_map']
        ind_dev, ind_val = folds_map[fold_num]
        if self.leak_enforcer.has_peekers:
            return np.append(ind_dev,ind_val), ind_val
        else:
            return ind_dev, ind_val

    def fit_transform(self, X, y, dataset_name):

        chain_of_chains = self.chain_of_chains

        assert X.shape[0] == len(y)

        if len(chain_of_chains) > 0:
            for chain in chain_of_chains:
                X_t, y_t = chain.fit_transform(X, y, dataset_name)
        else:
            X_t, y_t = X, y

        assert X_t.shape[0] == len(y_t)

        return X_t, y_t


    def transform(self,X, y, dataset_name):

        chain_of_chains = self.chain_of_chains

        if len(chain_of_chains) > 0:
            for chain in chain_of_chains:
                X_t, y_t = chain.transform(X, y, dataset_name)
        else:
            X_t, y_t = X, y

        return X_t, y_t

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

    def handle_missing_data(self, X, y):
        #TODO: Don't forget that you probably broke Horizontal Transformer combine.
        model_config = self.model_config
        project_settings = self.project_settings
        manipulations = model_config['feature_settings']['manipulations']
        try:
            first_manipulator_id = manipulations[0].keys()[0]
            first_manipulator_instance = manipulations[0][first_manipulator_id]['initialized_manipulator']
            if issubclass(first_manipulator_instance.__class__, Cleaner):
                has_cleaner = True
            else:
                has_cleaner = False
        except IndexError:
            has_cleaner = False
        le = self.leak_enforcer
        if not has_cleaner:
            delete_obs_entry = { 'auto.delete_obs': { 'inclusion_patterns' : 'All' }}
            model_config['feature_settings']['manipulations'] = [delete_obs_entry] + manipulations
            auto_delete_tc = TransformChain('auto_delete_tc', [delete_obs_entry],
                                            model_config, project_settings)
            le.update_manipulations([delete_obs_entry])
            auto_delete_tc.set_leak_enforcer(le)
            X, y = auto_delete_tc.fit_transform(X, y, 'train')
        else:
            i = self._find_last_cleaner(manipulations)
            cleaners = manipulations[0:i]
            cleaners_tc = TransformChain('cleaners_tc', cleaners, model_config, project_settings)
            le.update_manipulations(manipulations)
            cleaners_tc.set_leak_enforcer(le)
            X, y = cleaners_tc.fit_transform(X, y, 'train')
        return X, y

    def _find_last_cleaner(self, manipulations):
        i = 0
        manipulator_id = manipulations[i].keys()[0]
        manipulator_instance = manipulations[i][manipulator_id]['initialized_manipulator']
        while issubclass(manipulator_instance.__class__, Cleaner):
            i += 1
            try:
                manipulator_id = manipulations[i].keys()[0]
            except IndexError:
                return i
            manipulator_instance = manipulations[i][manipulator_id]['initialized_manipulator']
        return i

class LeakEnforcer:

    def __init__(self, manipulations_list, model_config):
        manipulator_map = dict()
        has_peekers = False
        for i in range(len(manipulations_list)):
            manipulator_entry = manipulations_list[i]
            manipulator_name = manipulator_entry.keys()[0]
            initialized_manipulator = manipulator_entry[manipulator_name]['initialized_manipulator']
            manipulator_peeking_status = initialized_manipulator.validation_peeking
            manipulator_map[manipulator_name] = { 'initialized_manipulator' : initialized_manipulator, #TODO: Why did I put initialized manipulator here?
                                                  'manipulator_peeking_status' : manipulator_peeking_status
                                                }
        if True in [item['manipulator_peeking_status'] for item in manipulator_map.values()]:
            has_peekers = True
        self.model_config = model_config
        self.manipulator_map = manipulator_map
        self.has_peekers = has_peekers

    def check_for_leak(self,X_mat):
        model_config = self.model_config
        if model_config.has_key('folds_map'):
            manipulator_map = self.manipulator_map
            if len(manipulator_map) > 0:
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
            else:
                return False
        else:
            return False

    def check_leak_allowed(self,manipulator_name):
        manipulator_map = self.manipulator_map
        return manipulator_map[manipulator_name]['manipulator_peeking_status']

    def update_manipulations(self, updated_manipulations):
        manipulator_map = self.manipulator_map
        for manipulator_entry in updated_manipulations:
            manipulator_id = manipulator_entry.keys()[0]
            initialized_manipulator = manipulator_entry[manipulator_id]['initialized_manipulator']
            manipulator_peeking_status = initialized_manipulator.validation_peeking
            if manipulator_id not in manipulator_map.keys():
                manipulator_map[manipulator_id] = { 'initialized_manipulator': initialized_manipulator,
                                                    'manipulator_peeking_status': manipulator_peeking_status }
        if True in [item['manipulator_peeking_status'] for item in manipulator_map.values()]:
            updated_self_peekers = True
            self.has_peekers = updated_self_peekers
        else:
            pass
        self.manipulator_map = manipulator_map

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
