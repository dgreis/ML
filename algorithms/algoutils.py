import importlib

from algorithms.wrapper import Wrapper


def configure_algorithms(model_configs, project_settings):
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