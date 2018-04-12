import importlib

def load_evaluation_battery(project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    if ml_problem_type == 'binary-classification':
        battery_keys = ['binary-classification','multiclass-classification']
    else:
        battery_keys =['multiclass-classification']
    battery_settings = dict()
    for key in battery_keys:
        ind_battery_settings = project_settings['metrics'][key]['battery']
        battery_settings.update(ind_battery_settings)
    battery = dict()
    #sklearn_metric_module = importlib.import_module('sklearn.metrics')
    for name in battery_settings:
        battery[name] = {
            'class' : load_metric_class(name),
            'kwargs' : battery_settings[name]['kwargs'],
            'metric_type' : battery_settings[name]['metric_type']
        }
    return battery

def load_metric_class(metric_name):
    sklearn_metric_module = importlib.import_module('sklearn.metrics')
    return getattr(sklearn_metric_module,metric_name)
