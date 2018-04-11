import yaml
import importlib

from chef import Chef
from utils import *
from evaluation import *
from cross_validation import *
from report import Report

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():
    chef = Chef()
    if not all_clean_input_files_exist(project_settings):
        prep_data = importlib.import_module('projects.' + project_settings['project_name'] + '.src.' + 'prep_data')
        prep_data.main()
        print "Data Built for project: " + global_settings['current_project']
    project_settings['working_files'] = project_settings['clean_input_files'].copy()

    model_configs = load_model_configs(project_settings)
    models = chef.configure_models(model_configs, project_settings)
    num_models = len(models)

    print "\nFit ML models for project: " + project_settings['project_name']
    print "Number of models to fit: " + str(num_models)
    i = 1
    report = Report(project_settings)
    for model_name in models:
        print "\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name +'. \nFirst Step: Finalize dataset'
        model_config = model_configs['Models'][model_name]
        model_config['model_name'] = model_name
        data = chef.prepare_final_model_dataset(model_config, project_settings)
        X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
        print "Data Finalized. Training data with " + str(len(X_train)) + " samples. Test data with " + str(len(X_test)) + \
              " samples and " + str(X_train.shape[1]) + " features."
        model = models[model_name]
        if model_config['cross_validation_settings'] != None:
            validator = Cross_Validator(model_config)
            validator.perform_cross_validation(X_train, y_train, model, model_config)
            model = validator.set_optimal_hyperparams(model)
        print "Next Step: Fit Model"
        model.fit(X_train,y_train)
        if model.gen_output_flag == True:
            model.gen_output()
        assert 1 == 1
        print 'Model Fit. Next Step: Perform Model Evaluation'
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

if __name__ == "__main__":
    main()