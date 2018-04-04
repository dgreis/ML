import importlib

def load_evaluation_battery(project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    if ml_problem_type == 'multiclass-classification':
         evaluation_battery = load_multiclass_eval_battery(project_settings)
    elif ml_problem_type == 'binary-classification':
        evaluation_battery = load_binaryclass_eval_metrics(project_settings) #TODO
    else:
        evaluation_battery = load_regression_eval_metrics(project_settings) #TODO
    return evaluation_battery

def load_multiclass_eval_battery(project_settings):
    battery_settings = project_settings['metrics']['classification']['multiclass']['battery']
    sklearn_metric_module = importlib.import_module('sklearn.metrics')
    eval_battery = dict()
    for name in battery_settings:
        eval_battery[name] = {
            'class' : getattr(sklearn_metric_module,name),
            'kwargs' : battery_settings[name]['kwargs']
        }
    return eval_battery