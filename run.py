import yaml
import os
import importlib

from utils import *
from evaluation import *

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

if not all_final_files_exist(project_settings):
    prep_data = importlib.import_module(project_settings['project_name'] + '.src.' + 'prep_data')
    prep_data.main()

model_configs = load_model_configs(project_settings)
models = configure_models(model_configs, project_settings)
num_models = len(models)

print "\nFit ML models for project: " + project_settings['project_name']
print "Number of models to fit: " + str(num_models)
i = 1
for model_name in models:
    print "\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name +'. \nStep One: Finalize dataset'
    data = prepare_final_model_dataset(model_configs['Models'][model_name], project_settings)
    print "Data Finalized. \nStep Two: Fit Model"
    X_train, y_train  = data['X_train'], data['y_train']
    model = models[model_name]
    model.fit(X_train,y_train)
    print 'Model Fit. \nStep Three: Perform Model Evaluation'
    X_test, y_test = data['X_test'], data['y_test']
    y_pred = model.predict(X_test)
    evaluation_battery = load_evaluation_battery(project_settings)
    report = dict()
    for metric_name in evaluation_battery:
        metric_class = evaluation_battery[metric_name]['class']
        kwargs = evaluation_battery[metric_name]['kwargs']
        report[metric_name] = metric_class(y_pred, y_test,**kwargs)
    i += 1

#m.label(y)
#score(m,y)

