import pandas as pd
import yaml
import os
import importlib

from feature.engineering import TransformChain
from feature.selection import FilterChain
from utils import *
from evaluation import *
from cross_validation import *
from report import Report
from utils import find_data_dir

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():
    if not all_clean_input_files_exist(project_settings):
        prep_data = importlib.import_module(project_settings['project_name'] + '.src.' + 'prep_data')
        prep_data.main()
    project_settings['working_files'] = project_settings['clean_input_files'].copy()

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
        model_config['model_name'] = model_name
        data = prepare_final_model_dataset(model_config, project_settings)
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


def prepare_final_model_dataset(model_config, project_settings):
    '''This presumes X_train, y_train, X_test, and y_test already exist. This
    method is meant to augment or exclude the base feature'''

    clean_input_files = project_settings['clean_input_files']

    X_mat_rel_filepaths = {x: clean_input_files[x] for x in ['X_train','X_test']}

    data_dir = find_data_dir(project_settings)

    data = dict()

    y_mat_rel_filepaths = {y: clean_input_files[y] for y in ['y_train','y_test']}

    for mat_name in y_mat_rel_filepaths:
        y_mat_file_path = data_dir + '/' + clean_input_files[mat_name]
        y_mat = pd.read_csv(y_mat_file_path,sep="\s+",engine='python',header=None)
        data[mat_name] = y_mat.iloc[:,0].tolist()

    X_train_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_train"]
    X_train = pd.read_csv(X_train_abs_filepath,sep="\s+",engine='python',header=None)
    y_train = data['y_train']

    transformations = model_config['feature_settings']['feature_engineering']
    filters = model_config['feature_settings']['feature_selection']

    if model_config['feature_settings']['select_before_eng']:
        fc = FilterChain(filters, model_config, project_settings)
        X_train_1st = fc.fit_transform(X_train, y_train)
        X_train_2nd = TransformChain(transformations, model_config, project_settings).fit_transform(X_train_1st)
    else:
        X_train_1st = TransformChain(transformations, model_config, project_settings).fit_transform(X_train)
        fc = FilterChain(filters, model_config, project_settings)
        X_train_2nd = fc.fit_transform(X_train_1st,y_train)

    data['X_train'] = X_train_2nd

    X_test_abs_filepath = data_dir + '/' + X_mat_rel_filepaths["X_test"]
    X_test = pd.read_csv(X_test_abs_filepath,sep="\s+",engine='python',header=None)
    if model_config['feature_settings']['select_before_eng']:
        X_test_1st = fc.transform(X_test,original_columns=True)
        X_test_2nd = TransformChain(transformations,model_config,project_settings,
                                    original_columns=True).fit_transform(X_test_1st,train=False)
    else:
        X_test_1st = TransformChain(transformations,model_config,project_settings,
                                    original_columns=True).fit_transform(X_test,train=False)
        X_test_2nd = fc.transform(X_test_1st)

    assert X_train_2nd.shape[1] == X_test_2nd.shape[1]

    data['X_test'] = X_test_2nd

    return data

if __name__ == "__main__":
    main()