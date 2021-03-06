import inspect

import pandas as pd
import yaml
import os
import collections

def find_data_dir(project_settings):
    abs_project_dir = find_project_dir(project_settings)
    rel_data_dir = project_settings['data_dir']
    abs_data_dir = abs_project_dir + '/' + rel_data_dir
    return abs_data_dir

def find_project_dir(project_settings):
    repo_loc = project_settings['repo_loc']
    project_name = project_settings['project_name']
    abs_project_dir = repo_loc + '/projects/' + project_name
    return abs_project_dir

def configure_project_settings(global_settings):
    repo_loc = global_settings['repo_loc']
    abs_project_dir = repo_loc + '/projects/' + global_settings['current_project']
    project_settings_loc = abs_project_dir + '/src/project_settings.yaml'
    project_settings = yaml.safe_load(open(project_settings_loc))
    new_settings = global_settings.copy()
    #new_settings.update(project_settings)
    new_settings = update(new_settings, project_settings)
    return new_settings

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_model_configs(project_settings):
    abs_project_dir = find_project_dir(project_settings)
    model_configs_loc = abs_project_dir + '/src/models.yaml'
    raw_model_configs = yaml.safe_load(open(model_configs_loc))
    model_configs = set_default_configs_if_missing(raw_model_configs, project_settings)
    return model_configs

def set_default_configs_if_missing(model_configs, project_settings):
    for model_name in model_configs['Models']:
        for facet in project_settings['model_facet_defaults']:
            if 'manipulations' in model_configs['Models'][model_name]:
                print("Put manipulations within 'feature settings' in models.yaml and re-run program")
                raise Exception
            if facet not in model_configs['Models'][model_name]:
                default = project_settings['model_facet_defaults'][facet]
                if type(default) == str:
                    model_configs['Models'][model_name][facet] = eval(default)
                else:
                    model_configs['Models'][model_name][facet] = default
            elif facet == 'feature_settings':
                default_feature_settings = project_settings['model_facet_defaults']['feature_settings']
                for sub_facet in default_feature_settings:
                    if not sub_facet in model_configs['Models'][model_name]['feature_settings']:
                        model_configs['Models'][model_name]['feature_settings'][sub_facet] = default_feature_settings[sub_facet]
        for facet in model_configs['Models'][model_name]:
            if facet not in project_settings['model_facet_defaults']:
                print(facet + " in model " + model_name + " is not correctly specified. Check defaults in global_settings.yaml and try again.")
                raise Exception
            else:
                pass
    return model_configs

def all_clean_input_files_exist(project_settings):
    data_dir = find_data_dir(project_settings)
    rel_filepaths = project_settings['clean_input_files'].values()
    abs_filepaths = [data_dir + '/' + fp for fp in rel_filepaths]
    for fp in abs_filepaths:
        if not os.path.isfile(fp):
            return False
    return True

def flip_dict(orig_dict):
    flipped_dict = {v:k for k,v in orig_dict.items()}
    return flipped_dict

def is_fully_qualified_path(project_settings, path):
    repo_loc = project_settings['repo_loc']
    if repo_loc in path:
        return True
    else:
        return False

def load_working_file_filepath(project_settings, data_ref):
    assert project_settings.has_key('working_files')
    working_files = project_settings['working_files']
    filepath = working_files[data_ref]
    if is_fully_qualified_path(project_settings, filepath):
        wf_abs_filepath = filepath
    else:
        data_dir = find_data_dir(project_settings)
        wf_abs_filepath = data_dir + '/' + filepath
    return wf_abs_filepath


def load_inv_column_map(filepath):
    df = pd.read_csv(filepath, sep="\s+", engine='python', names=['col_name'])
    return pd.Series(df.index, index=df.col_name).to_dict()


def load_clean_input_file_filepath(project_settings, data_ref):
    cif_rel_filepath = project_settings['clean_input_files'][data_ref]
    data_dir = find_data_dir(project_settings)
    cif_abs_filepath = data_dir + '/' + cif_rel_filepath
    return cif_abs_filepath

def det_num_cv_folds(model_config):
    if model_config['cv_num_folds'] < 2:
        try:
            assert 'train_val_split' in model_config
        except AssertionError:
            print("If not performing CV please specify train/val split with 'train_val_split' in yaml")
            raise Exception
    else:
        pass
    return model_config['cv_num_folds']

def finalize_manipulations(model_config, project_settings):
    if 'preprocess_config' in model_config['feature_settings']:
        preprocess_config_name = model_config['feature_settings']['preprocess_config']
        try:
            assert project_settings.has_key('preprocess_configs') and project_settings['preprocess_configs'].has_key(
                preprocess_config_name)
        except AssertionError:
            print("Preprocess configs not correctly specified")
            raise Exception
        preprocess_manipulations = [
            {'handle_missing_data': project_settings['preprocess_configs'][preprocess_config_name]}]
    else:
        preprocess_manipulations = list()

    manipulations = preprocess_manipulations + model_config['feature_settings']['manipulations']
    model_config['feature_settings']['manipulations'] = manipulations
    return model_config

def get_args(class_, method):
    target = getattr(class_,method)
    args = getattr(inspect.getargspec(target),'args')
    return args[1:]