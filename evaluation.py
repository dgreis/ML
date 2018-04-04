import importlib

def load_evaluation_battery(project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    battery_settings = project_settings['metrics'][ml_problem_type]['battery']
    battery = dict()
    sklearn_metric_module = importlib.import_module('sklearn.metrics')
    for name in battery_settings:
        battery[name] = {
            'class' : getattr(sklearn_metric_module,name),
            'kwargs' : battery_settings[name]['kwargs']
        }
    return battery