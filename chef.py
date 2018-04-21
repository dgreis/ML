import importlib

import pandas as pd

from feature.engineering import TransformChain
from feature.selection import FilterChain
from algorithms.wrapper import Wrapper
from utils import find_data_dir



class Chef:

    def __init__(self):
        pass

    def prepare_final_model_dataset(self, model_config, project_settings):
        '''This presumes X_train, y_train, X_test, and y_test already exist. This
        method is meant to augment or exclude the base feature'''

        clean_input_files = project_settings['clean_input_files']

        X_mat_rel_filepaths = {x: clean_input_files[x] for x in ['X_train','X_train_val','X_val','X_test']}

        data_dir = find_data_dir(project_settings)

        data = dict()

        y_mat_rel_filepaths = {y: clean_input_files[y] for y in ['y_train_val', 'y_test', 'y_val','y_train']}

        for mat_name in y_mat_rel_filepaths:
            y_mat_file_path = data_dir + '/' + clean_input_files[mat_name]
            y_mat = pd.read_csv(y_mat_file_path, sep="\s+", engine='python', header=None)
            data[mat_name] = y_mat.iloc[:, 0].tolist()

        X_train_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_train"]
        X_train = pd.read_csv(X_train_abs_filepath, sep="\s+", engine='python', header=None)
        y_train = data['y_train']
        y_train_val = data['y_train_val']

        transformations = model_config['feature_settings']['feature_engineering']
        filters = model_config['feature_settings']['feature_selection']

        if model_config['feature_settings']['select_before_eng']:
            fc = FilterChain(filters, model_config, project_settings)
            fc.fit(X_train, y_train)
            X_train_1st = fc.transform(X_train,dataset_name="Train")
            tc = TransformChain(transformations,model_config,project_settings)
            X_train_2nd = tc.transform(X_train_1st,y_train)
        else:
            tc = TransformChain(transformations,model_config,project_settings)
            X_train_1st = tc.transform(X_train,y_train)
            fc = FilterChain(filters, model_config, project_settings)
            fc.fit(X_train_1st, y_train)
            X_train_2nd = fc.transform(X_train_1st,dataset_name="Train")

        data['X_train'] = X_train_2nd

        X_val_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_val"]
        X_val = pd.read_csv(X_val_abs_filepath, sep="\s+", engine='python', header=None)
        y_val = data['y_val']
        if model_config['feature_settings']['select_before_eng']:
            X_val_1st = fc.transform(X_val, original_columns=True,dataset_name='Val')
            X_val_2nd = tc.transform(X_val_1st,y_val, dataset_name="Val")
        else:
            X_val_1st = tc.transform(X_val,y_val, dataset_name="Val")
            X_val_2nd = fc.transform(X_val_1st,dataset_name='Val')

        assert X_train_2nd.shape[1] == X_val_2nd.shape[1]

        data["X_val"] = X_val_2nd

        X_train_val_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_train_val"]
        X_train_val = pd.read_csv(X_train_val_abs_filepath, sep="\s+", engine='python', header=None)
        if model_config['feature_settings']['select_before_eng']:
            X_train_val_1st = fc.transform(X_train_val, original_columns=True,dataset_name="Train_val")
            X_train_val_2nd = tc.transform(X_train_val_1st,y_train_val, dataset_name="Train_val")
        else:
            X_train_val_1st = tc.transform(X_train_val,y_train_val,dataset_name="Train_val")
            X_train_val_2nd = fc.transform(X_train_val_1st,dataset_name="Train_val")

        assert X_train_2nd.shape[1] == X_train_val_2nd.shape[1]

        data['X_train_val'] = X_train_val_2nd

        X_test_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_test"]
        X_test = pd.read_csv(X_test_abs_filepath, sep="\s+", engine='python', header=None)
        y_test = data['y_test']
        if model_config['feature_settings']['select_before_eng']:
            X_test_1st = fc.transform(X_test, original_columns=True,dataset_name="Test")
            X_test_2nd = tc.transform(X_test_1st,y_test, dataset_name="Test")
        else:
            X_test_1st = tc.transform(X_test, y_test, dataset_name="Test")
            X_test_2nd = fc.transform(X_test_1st,dataset_name='Test')

        assert X_train_2nd.shape[1] == X_test_2nd.shape[1]

        data['X_test'] = X_test_2nd

        return data

    def configure_models(self, model_configs, project_settings):
        models = model_configs['Models']
        algos = dict()
        for model_name in models:
            model_config = models[model_name]
            model_config['model_name'] = model_name
            module_comps = models[model_name]['base_algorithm'].split('.')
            module_name = ('.').join(module_comps[:-1])
            module = importlib.import_module(module_name)
            base_algo_class_name = module_comps[-1:][0]
            base_algo_class = getattr(module, base_algo_class_name)
            # kwargs = models[model_name]['keyword_arg_settings']
            # other_options = models[model_name]['other_options']
            if 'algorithms' not in module_name:
                base_algo_instance = Wrapper(base_algo_class, model_config, project_settings)
            else:
                base_algo_instance = base_algo_class(model_config, project_settings)
            algos[model_name] = base_algo_instance
        return algos