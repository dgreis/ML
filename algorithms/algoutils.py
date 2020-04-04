import importlib

from algorithms.wrapper import Wrapper

def configure_algorithm(model_config, project_settings):
    base_algorithm = model_config['base_algorithm']
    base_algo_class = get_algo_class(base_algorithm)
    if ('.').join([base_algo_class.__module__,base_algo_class.__name__]) in project_settings['overwritten_algos']:
        print("Don't use original " + base_algorithm + " sklearn implementation. Check algorithms in package for custom package implementation instead")
        raise Exception
    if 'algorithms' not in base_algorithm:
        base_algo_instance = Wrapper('final_algorithm', base_algo_class, model_config, project_settings)  #Wrapper is used for algos implemented by others, i.e. sklearn
    else:
        base_algo_instance = base_algo_class('final_algorithm', model_config, project_settings)
    return base_algo_instance

def get_algo_class(base_algorithm_name):
    base_algorithm = max(base_algorithm_name.split('__'), key=len)
    module_comps = base_algorithm.split('.')
    module_name = ('.').join(module_comps[:-1])
    module = importlib.import_module(module_name)
    base_algo_class_name = module_comps[-1:][0]
    base_algo_class = getattr(module, base_algo_class_name)
    return base_algo_class
