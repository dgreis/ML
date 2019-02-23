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
    project_settings = yaml.load(open(project_settings_loc))
    new_settings = global_settings.copy()
    #new_settings.update(project_settings)
    new_settings = update(new_settings, project_settings)
    return new_settings

def update(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_model_configs(project_settings):
    abs_project_dir = find_project_dir(project_settings)
    model_configs_loc = abs_project_dir + '/src/models.yaml'
    raw_model_configs = yaml.load(open(model_configs_loc))
    model_configs = set_default_configs_if_missing(raw_model_configs, project_settings)
    return model_configs

def set_default_configs_if_missing(model_configs, project_settings):
    for model_name in model_configs['Models']:
        for facet in project_settings['model_facet_defaults']:
            if not model_configs['Models'][model_name].has_key(facet):
                default = project_settings['model_facet_defaults'][facet]
                if type(default) == str:
                    model_configs['Models'][model_name][facet] = eval(default)
                else:
                    model_configs['Models'][model_name][facet] = default
            elif facet == 'feature_settings':
                default_feature_settings = project_settings['model_facet_defaults']['feature_settings']
                for sub_facet in default_feature_settings:
                    if not model_configs['Models'][model_name]['feature_settings'].has_key(sub_facet):
                        model_configs['Models'][model_name]['feature_settings'][sub_facet] = default_feature_settings[sub_facet]
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
    flipped_dict = {v:k for k,v in orig_dict.iteritems()}
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

def det_num_cv_folds(model_config, project_settings):
    if model_config.has_key('cv_num_folds'):
        if model_config['cv_num_folds'] < 2:
            try:
                assert model_config.has_key('train_val_split')
            except AssertionError:
                print "If not performing CV please specify train/val split with 'train_val_split' in yaml"
                raise Exception
            #train_val_split = model_config['train_val_split']
            #implied_folds = float.as_integer_ratio(train_val_split)[1]
            #assert implied_folds <= 10
            #return implied_folds
        else:
            return model_config['cv_num_folds']
    else:
        return project_settings['assessment']['cv_num_folds']
