import importlib
import yaml
import os

from algorithms.wrapper import Wrapper

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
    ml_problem_type = project_settings['ml_problem_type']
    eval_pak = global_settings['metrics'][ml_problem_type]
    battery = eval_pak['battery']
    for metric in battery:
        if not battery[metric].has_key('kwargs'):
            battery[metric]['kwargs'] = eval_pak['standard_keyword_args']
    new_settings = global_settings.copy()
    new_settings.update(project_settings)
    return new_settings

def fetch_eval_pak(project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    eval_pak = project_settings['metrics'][ml_problem_type]
    return eval_pak

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

def configure_models(model_config):
    models = model_config['Models']
    algos = dict()
    for model_name in models:
        module_comps = models[model_name]['base_algorithm'].split('.')
        module_name = ('.').join(module_comps[:-1])
        module = importlib.import_module(module_name)
        base_algo_class_name = module_comps[-1:][0]
        base_algo_class = getattr(module,base_algo_class_name)
        kwargs = models[model_name]['keyword_arg_settings']
        other_options = models[model_name]['other_options']
        if 'algorithms' not in module_name:
            base_algo_instance = Wrapper(base_algo_class, other_options, **kwargs)
        else:
            base_algo_instance = base_algo_class(other_options,**kwargs)
        algos[model_name] = base_algo_instance
    return algos


def all_clean_input_files_exist(project_settings):
    data_dir = find_data_dir(project_settings)
    rel_filepaths = project_settings['clean_input_files'].values()
    abs_filepaths = [data_dir + '/' + fp for fp in rel_filepaths]
    for fp in abs_filepaths:
        if not os.path.isfile(fp):
            return False
    return True


