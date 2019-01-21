import importlib
import numpy as np

def load_evaluation_battery(project_settings):
    ml_problem_type = project_settings['ml_problem_type']
    if ml_problem_type == 'binary-classification':
        battery_keys = ['binary-classification','multiclass-classification']
    elif ml_problem_type == 'multiclass-classification':
        battery_keys =['multiclass-classification']
    elif ml_problem_type == 'regression':
        battery_keys = ['regression']
    else:
        raise NotImplementedError
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
    if metric_name != "rmsle":
        sklearn_metric_module = importlib.import_module('sklearn.metrics')
        return getattr(sklearn_metric_module,metric_name)
    else:
        this_module = importlib.import_module('evaluation')
        return getattr(this_module,metric_name)


def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            return np.nan
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5