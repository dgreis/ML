import importlib

def find_data_dir(global_settings,project_settings):
    repo_loc = global_settings['repo_loc']
    project_dir = project_settings['project_name']
    rel_data_dir = project_settings['data_dir']
    abs_data_dir = repo_loc + '/' + project_dir + '/' + rel_data_dir
    return abs_data_dir

def build_algorithms(algorithm_config,project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    ml_module = importlib.import_module('algorithms.' + ml_problem_type )
    models = algorithm_config['Algorithms']
    algos = list()
    for model_name in models:
        algo_class = getattr(ml_module,models[model_name]['base_algorithm'])
        kwargs = models[model_name]['keyword_arg_settings']
        algo_instance = algo_class(**kwargs)
        algos.append(algo_instance)
    return algos

