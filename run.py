from __future__ import print_function

import fractions
import copy

from utils import *
from evaluation import *
from cross_validation import *
from report import Report
from algorithms.algoutils import configure_algorithms

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

#TODO: some sort of clean-up of feature files so that program doesn't mistakenly use from past runs
def main():
    if not all_clean_input_files_exist(project_settings):
        prep_data = importlib.import_module('projects.' + project_settings['project_name'] + '.src.' + 'prep_data')
        prep_data.main()
        print("Data Built for project: " + global_settings['current_project'])
    project_settings['working_files'] = project_settings['clean_input_files'].copy()

    model_configs = load_model_configs(project_settings)
    models = configure_algorithms(model_configs, project_settings)
    num_models = len(models)

    print("\nFit ML models for project: " + project_settings['project_name'])
    print( "Number of models to fit: " + str(num_models))
    i = 1
    report = Report(project_settings)
    for model_name in models:
        print("\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name)
        model_config = model_configs['Models'][model_name]
        model_config['model_name'] = model_name
        if project_settings.has_key('numeric_features'):
            model_config['numeric_features'] = copy.copy(project_settings['numeric_features'])
            #cleaner to make new one for each model
        manager = Manager(model_config,project_settings)
        data = manager.load_clean_datasets('train_val',project_settings)
        model = models[model_name]
        cv = CrossValidator(model_config, project_settings)
        cv_num_folds = det_num_cv_folds(model_config, project_settings)
        if cv_num_folds > 1:
            print("Next Step: estimate generalization error for some metrics using CV")
            entries = cv.perform_cross_validation(data, model)
            report.add_entries_to_report(entries)
        else:
            print("CV folds set to less than 2. No CV performed. ",end="")
        #TODO: Figure out 'remaining metrics' apply to regression. If not, this part could be skipped. Answer: Yes, only regression
        print("Next Step: estimate remaining metrics using validation set.")
        X_train_val, y_train_val = manager.load_clean_datasets('train_val', project_settings)['train_val']
        if cv_num_folds < 2:
            implied_folds = fractions.Fraction.from_float(model_config['train_val_split']).limit_denominator(10).denominator
            assert implied_folds <= 10
            folds_map = cv.gen_folds_map(X_train_val, y_train_val, implied_folds)  #TODO: Maybe make me specify if I don't do CV
            model_config['folds_map'] = folds_map
            model_config['fold_i'] = implied_folds - 1
        ind_dev, ind_val = manager.return_fold_dev_val_ind(model_config['fold_i'])
        X_train, y_train = X_train_val.loc[ind_dev, :], pd.Series(y_train_val).loc[ind_dev].tolist()
        X_train_p, y_train_p = manager.fit_transform(X_train, y_train, 'train')
        X_val, y_val = X_train_val.loc[ind_val, :], pd.Series(y_train_val).loc[ind_val].tolist()
        X_val_p, y_val_p = manager.transform(X_val, y_val, 'val')
        assert X_train_p.shape[1] == X_val_p.shape[1]
        assert len(y_val_p) == len(y_val)
        le = manager.leak_enforcer
        if le.check_for_leak(X_train_p):
            X_train_p, y_train_p = le.remove_leaking_indices(X_train_p, y_train_p)
        print("Validation dataset finalized. Training with " + str(len(X_train_p)) + " samples. Validation data with " +\
              str(len(X_val_p)) + " samples. Model with " + str(X_val_p.shape[1]) + " features. Now fitting model... ",end="")
        model.fit(X_train_p, y_train_p)
        print('Model Fit.\nNext Step: Perform Model Evaluation')
        y_pred = model.predict(X_val_p)
        if model.gen_output_flag:
            model.gen_output()
        evaluation_battery = load_evaluation_battery(project_settings)
        #non_cv_battery = {k: v for k, v in evaluation_battery.items() if evaluation_battery[k]['metric_type'] != 'column'}
        report_row = dict()
        for metric_name in evaluation_battery:
            report_row['dataset_name'] = 'validation'
            report_row['model_name'] = model_config['model_name']
            if hasattr(cv,'optimal_hyperparams'):
                report_row['setting'] = str(cv.optimal_hyperparams)
            else:
                report_row['setting'] = 'default'
            report_row['metric'] = metric_name
            metric_class = evaluation_battery[metric_name]['class']
            kwargs = evaluation_battery[metric_name]['kwargs']
            try:
                content = metric_class(y_pred, y_val,**kwargs)
            except ValueError:
                content = np.nan
            report_row['content'] = content
            report.add_entries_to_report(report_row)
        if model_config['evaluate_testset']:
            #TODO: implement this
            X_test,y_test = manager.load_clean_datasets('test',project_settings)['test']
            X_test_p,y_test_p = manager.transform(X_test,y_test,'test')
            assert y_test == y_test_p
            y_pred = model.predict(X_test_p)
            for metric_name in evaluation_battery:
                report_row['dataset_name'] = 'test'
                report_row['model_name'] = model_config['model_name']
                if hasattr(cv, 'optimal_hyperparams'):
                    report_row['setting'] = str(cv.optimal_hyperparams)
                else:
                    report_row['setting'] = 'default'
                report_row['metric'] = metric_name
                metric_class = evaluation_battery[metric_name]['class']
                kwargs = evaluation_battery[metric_name]['kwargs']
                #try: next line was indented
                content = metric_class(y_pred, y_test_p, **kwargs)
                #except ValueError:
                #    content = np.nan
                report_row['content'] = content
                report.add_entries_to_report(report_row)
        i += 1
    print("All Models Fit and Evaluated. Writing Report")
    report.write_report()
    print("\nReport written to: " + project_settings['repo_loc'] +'/' + project_settings['project_name'] + '/evaluation/report.html')

if __name__ == "__main__":
    main()