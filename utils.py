import importlib
import yaml
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
    return yaml.load(open(model_configs_loc))

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

    for mat_name in X_mats:
        abs_filepath = data_dir + '/' + X_mats[mat_name]
        X_mat = pd.read_csv(abs_filepath,sep="'\s+",engine='python') #TODO: This isn't parsing correctly again. Also yaml issue

    feature_settings = model_config['feature_settings']
