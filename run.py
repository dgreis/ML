import yaml
import os
import importlib

from utils import *
from evaluation import *
from cross_validation import *
from report import Report

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
report = Report(project_settings)
for model_name in models:
    print "\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name +'. \nFirst Step: Finalize dataset'
    model_config = model_configs['Models'][model_name]
    data = prepare_final_model_dataset(model_config, project_settings)
    print "Data Finalized"
    X_train, y_train  = data['X_train'], data['y_train']
    model = models[model_name]
    if model_config['cross_validation_settings'] != None:
        validator = Cross_Validator(model_config)
        validator.perform_cross_validation(X_train, y_train, model, model_config)
        model = validator.set_optimal_hyperparams(model)
    print "Next Step: Fit Model"
    model.fit(X_train,y_train)
    print 'Model Fit. \nNext Step: Perform Model Evaluation'
    X_test, y_test = data['X_test'], data['y_test']
    y_pred = model.predict(X_test)
    evaluation_battery = load_evaluation_battery(project_settings)
    for metric_name in evaluation_battery:
        metric_class = evaluation_battery[metric_name]['class']
        kwargs = evaluation_battery[metric_name]['kwargs']
        report.add_entry(metric_name, model_name, metric_class(y_pred, y_test,**kwargs))
    i += 1
print "All Models Fit and Evaluated. Writing Report"
report.write_report()
print "\nReport written to: " + project_settings['repo_loc'] +'/' + project_settings['project_name'] + '/evaluation/report.html'

