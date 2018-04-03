import importlib
import yaml
import os
import pandas as pd

global_settings = yaml.load(open('./global_settings.yaml'))

def find_data_dir(global_settings,project_settings):
    repo_loc = global_settings['repo_loc']
    project_dir = project_settings['project_name']
    rel_data_dir = project_settings['data_dir']
    abs_data_dir = repo_loc + '/' + project_dir + '/' + rel_data_dir
    return abs_data_dir

def load_project_settings(global_settings):
    repo_loc = global_settings['repo_loc']
    project_dir = repo_loc + '/' + global_settings['current_project']
    project_settings_loc = project_dir + '/src/project_settings.yaml'
    return yaml.load(open(project_settings_loc))

def load_model_configs(global_settings):
    repo_loc = global_settings['repo_loc']
    project_dir = repo_loc + '/' + global_settings['current_project']
    model_configs_loc = project_dir + '/src/models.yaml'
    raw_model_configs = yaml.load(open(model_configs_loc))
    model_configs = set_default_configs_if_missing(raw_model_configs,global_settings)
    return model_configs

def set_default_configs_if_missing(model_configs, global_settings):
    for model_name in model_configs['Models']:
        for facet in global_settings['model_facet_defaults']:
            if not model_configs['Models'][model_name].has_key(facet):
                model_configs[model_name][facet] = global_settings['model_facet_defaults'][facet]
    return model_configs

def configure_models(algorithm_config, project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    ml_module = importlib.import_module('algorithms.' + ml_problem_type )
    models = algorithm_config['Models']
    algos = dict()
    for model_name in models:
        algo_class = getattr(ml_module,models[model_name]['base_algorithm'])
        kwargs = models[model_name]['keyword_arg_settings']
        algo_instance = algo_class(**kwargs)
        algos[model_name] = algo_instance
    return algos


def prepare_final_model_dataset(model_config, project_settings):
    '''This presumes X_train, y_train, X_test, and y_test already exist. This
    method is meant to augment or exclude the base features'''

    final_files = project_settings['final_files']

    X_mats = {x: final_files[x] for x in ['X_train','X_test']}

    data_dir = find_data_dir(global_settings, project_settings)

    feature_settings = model_config['feature_settings']

    column_names_filepath = data_dir + '/' + project_settings['final_files']['feature_names']
    inv_column_map = load_column_map(column_names_filepath)

    data = dict()

    for mat_name in X_mats:
        abs_X_filepath = data_dir + '/' + X_mats[mat_name]
        X_mat = pd.read_csv(abs_X_filepath,sep="\s+",engine='python',header=None) #TODO: This isn't parsing correctly again. Also yaml issue
        X_mat_filt = filter_columns(X_mat,feature_settings,inv_column_map)
        X_mat_transform = transform_columns(X_mat_filt,feature_settings)
        data[mat_name] = X_mat_transform

    y_mats = {y: final_files[y] for y in ['y_train','y_test']}

    for mat_name in y_mats:
        y_mat_file_path = data_dir + '/' + final_files[mat_name]
        y_mat = pd.read_csv(y_mat_file_path,sep="\s+",engine='python',header=None)
        data[mat_name] = y_mat.iloc[:,0].tolist()

    return data


def all_final_files_exist(project_settings,global_settings):
    data_dir = find_data_dir(global_settings,project_settings)
    rel_filepaths = project_settings['final_files']
    abs_filepaths = [data_dir + '/' + fp for fp in rel_filepaths]
    for fp in abs_filepaths:
        if not os.path.isfile(fp):
            return False
    return True

def load_column_map(filepath):
    df = pd.read_csv(filepath,sep="\s+",engine='python',names=['col_name'])
    return pd.Series(df.index,index=df.col_name).to_dict()

def transform_columns(X_mat,feature_settings):
    return X_mat

def filter_columns(X_mat,feature_settings,inv_column_map):
    if feature_settings['exclusion_patterns'] == 'None':
        return X_mat
    else:
        exclude_columns = list()
        col_names = inv_column_map.keys()
        for pattern in feature_settings['exclusion_patterns']:
            pat_exclude_columns = filter(lambda x: pattern in x, col_names)
            exclude_columns = exclude_columns + pat_exclude_columns
        exclude_indices = [int(inv_column_map[col_name]) for col_name in exclude_columns]
        X_mat_filt = X_mat.drop(axis=1,labels=exclude_indices)
    return X_mat_filt


            


