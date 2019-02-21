import pandas as pd

from feature.engineering import TransformChain
from feature.selection import FilterChain
from utils import find_data_dir, load_clean_input_file_filepath, load_inv_column_map, flip_dict



class Manager:

    def __init__(self,model_config, project_settings):
        self.model_config = model_config
        self.project_settings = project_settings

    def return_fold_dev_val_ind(self,fold_num):
        model_config = self.model_config
        #TODO: If model will peek at validation set, implement logic here
        folds_map = model_config['folds_map']
        ind_dev, ind_val = folds_map[fold_num]
        return ind_dev, ind_val

    def fit_transform(self, X, y, dataset_name):

        model_config = self.model_config
        project_settings = self.project_settings

        transformations = model_config['feature_settings']['feature_engineering']
        filters = model_config['feature_settings']['feature_selection']

        if model_config['feature_settings']['select_before_eng']:
            fc = FilterChain(filters, model_config, project_settings)
            X_1st, y_1st = fc.fit_transform(X, y, dataset_name)
            tc = TransformChain(transformations, model_config, project_settings)
            X_2nd, y_2nd = tc.fit_transform(X_1st, y_1st,dataset_name)
        else:
            tc = TransformChain(transformations, model_config, project_settings)
            X_1st, y_1st = tc.fit_transform(X, y,dataset_name)
            fc = FilterChain(filters, model_config, project_settings)
            X_2nd, y_2nd = fc.fit_transform(X_1st, y_1st, dataset_name)

        assert X.shape[0] == len(y)

        self.fc = fc
        self.tc = tc

        return X_2nd, y_2nd


    def transform(self,X, y, dataset_name):

        model_config = self.model_config
        project_settings = self.project_settings

        fc = self.fc
        tc = self.tc

        if model_config['feature_settings']['select_before_eng']:
            X_1st, y_1st = fc.transform(X,y,dataset_name)
            X_2nd, y_2nd = tc.transform(X_1st,y_1st, dataset_name)
            total_manipulations = fc.filters + tc.transformations
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

    # def prepare_final_model_dataset(self, model_config, project_settings):
    #     '''This presumes X_train, y_train, X_test, and y_test already exist. This
    #     method is meant to augment or exclude the base feature'''
    #
    #     clean_input_files = project_settings['clean_input_files']
    #
    #     X_mat_rel_filepaths = {x: clean_input_files[x] for x in ['X_train','X_train_val','X_val','X_test']}
    #
    #     data_dir = find_data_dir(project_settings)
    #
    #     data = dict()
    #
    #     y_mat_rel_filepaths = {y: clean_input_files[y] for y in ['y_train_val', 'y_test', 'y_val','y_train']}
    #
    #     for mat_name in y_mat_rel_filepaths:
    #         y_mat_file_path = data_dir + '/' + clean_input_files[mat_name]
    #         y_mat = pd.read_csv(y_mat_file_path, sep="\s+", engine='python', header=None)
    #         data[mat_name] = y_mat.iloc[:, 0].tolist()
    #
    #     X_train_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_train"]
    #     X_train = pd.read_csv(X_train_abs_filepath, sep="\s+", engine='python', header=None)
    #     y_train = data['y_train']
    #     y_train_val = data['y_train_val']
    #
    #     transformations = model_config['feature_settings']['feature_engineering']
    #     filters = model_config['feature_settings']['feature_selection']
    #
    #     if model_config['feature_settings']['select_before_eng']:
    #         fc = FilterChain(filters, model_config, project_settings)
    #         fc.fit(X_train, y_train)
    #         X_train_1st,y_train_1st = fc.transform(X_train,dataset_name="Train")
    #         tc = TransformChain(transformations,model_config,project_settings)
    #         X_train_2nd, y_train_2nd = tc.fit_transform(X_train_1st,y_train_1st)
    #     else:
    #         tc = TransformChain(transformations,model_config,project_settings)
    #         X_train_1st, y_train_1st = tc.fit_transform(X_train,y_train)
    #         fc = FilterChain(filters, model_config, project_settings)
    #         fc.fit(X_train_1st, y_train_1st)
    #         X_train_2nd, y_train_2nd = fc.transform(X_train_1st,y_train_1st,dataset_name="Train")
    #
    #     assert X_train.shape[0] == len(y_train)
    #
    #     data['X_train'] = X_train_2nd
    #     data['y_train'] = y_train_2nd
    #
    #     X_val_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_val"]
    #     X_val = pd.read_csv(X_val_abs_filepath, sep="\s+", engine='python', header=None)
    #     y_val = data['y_val']
    #     if model_config['feature_settings']['select_before_eng']:
    #         X_val_1st, y_val_1st = fc.transform(X_val,y_val,original_columns=True,dataset_name='Val')
    #         X_val_2nd, y_val_2nd = tc.transform(X_val_1st,y_val_1st, dataset_name="Val")
    #     else:
    #         X_val_1st, y_val_1st = tc.transform(X_val,y_val, dataset_name="Val")
    #         X_val_2nd, y_val_2nd = fc.transform(X_val_1st,y_val_1st, dataset_name='Val')
    #
    #     assert X_train_2nd.shape[1] == X_val_2nd.shape[1]
    #     #assert X_val_2nd.shape[0] == len(y_val_2nd)
    #
    #     data["X_val"] = X_val_2nd
    #     data["y_val"] = y_val_2nd
    #
    #     X_train_val_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_train_val"]
    #     X_train_val = pd.read_csv(X_train_val_abs_filepath, sep="\s+", engine='python', header=None)
    #     if model_config['feature_settings']['select_before_eng']:
    #         X_train_val_1st,y_train_val_1st = fc.transform(X_train_val,y_train_val,original_columns=True,dataset_name="Train_val")
    #         X_train_val_2nd, y_train_val_2nd = tc.transform(X_train_val_1st,y_train_val_1st, dataset_name="Train_val")
    #     else:
    #         X_train_val_1st, y_train_val_1st = tc.transform(X_train_val,y_train_val,dataset_name="Train_val")
    #         X_train_val_2nd, y_train_val_2nd = fc.transform(X_train_val_1st,y_train_val_1st,dataset_name="Train_val")
    #
    #     assert X_train_2nd.shape[1] == X_train_val_2nd.shape[1]
    #     assert X_train_val_2nd.shape[0] == len(y_train_val_2nd)
    #
    #     data['X_train_val'] = X_train_val_2nd
    #     data['y_train_val'] = y_train_val_2nd
    #
    #     X_test_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_test"]
    #     X_test = pd.read_csv(X_test_abs_filepath, sep="\s+", engine='python', header=None)
    #     y_test = data['y_test']
    #     if model_config['feature_settings']['select_before_eng']:
    #         X_test_1st, y_test_1st = fc.transform(X_test,y_test,original_columns=True,dataset_name="Test")
    #         X_test_2nd, y_test_2nd = tc.transform(X_test_1st,y_test_1st, dataset_name="Test")
    #     else:
    #         X_test_1st, y_test_1st = tc.transform(X_test, y_test, dataset_name="Test")
    #         X_test_2nd, y_test_2nd = fc.transform(X_test_1st,y_test_1st,dataset_name='Test')
    #
    #     assert X_train_2nd.shape[1] == X_test_2nd.shape[1]
    #
    #     data['X_test'] = X_test_2nd
    #     data['y_test'] = y_test_2nd
    #
    #     return data

