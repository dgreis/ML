import importlib

from algorithms.wrapper import Wrapper


def configure_algorithms(model_configs, project_settings):
    models = model_configs['Models']
    algos = dict()
    for model_name in models:
        model_config = models[model_name]
        model_config['model_name'] = model_name
        base_algorithm = models[model_name]['base_algorithm']
        base_algo_class = get_algo_class(base_algorithm)
        if 'algorithms' not in base_algorithm:
            base_algo_instance = Wrapper(base_algo_class, model_config, project_settings) #Wrapper is used for algos implemented by others, i.e. sklearn
        else:
            base_algo_instance = base_algo_class(model_config, project_settings)
        algos[model_name] = base_algo_instance
    return algos

def get_algo_class(base_algorithm):
    module_comps = base_algorithm.split('.')
    module_name = ('.').join(module_comps[:-1])
    module = importlib.import_module(module_name)
    base_algo_class_name = module_comps[-1:][0]
    base_algo_class = getattr(module, base_algo_class_name)
    return base_algo_class
