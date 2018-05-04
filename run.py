from utils import *
from evaluation import *
from cross_validation import *
from report import Report
from algorithms.algoutils import configure_algorithms

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():
    if not all_clean_input_files_exist(project_settings):
        prep_data = importlib.import_module('projects.' + project_settings['project_name'] + '.src.' + 'prep_data')
        prep_data.main()
        print "Data Built for project: " + global_settings['current_project']
    project_settings['working_files'] = project_settings['clean_input_files'].copy()

    model_configs = load_model_configs(project_settings)
    models = configure_algorithms(model_configs, project_settings)
    num_models = len(models)

    print "\nFit ML models for project: " + project_settings['project_name']
    print "Number of models to fit: " + str(num_models)
    i = 1
    report = Report(project_settings)
    for model_name in models:
        print "\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name
        model_config = model_configs['Models'][model_name]
        model_config['model_name'] = model_name
        manager = Manager(model_config,project_settings)
        data = manager.load_clean_datasets('train_val',project_settings)
        model = models[model_name]
        cv = CrossValidator(model_config, project_settings)
        print "Next Step: estimate generalization error for some metrics using CV"
        entries = cv.perform_cross_validation(data, model)
        #averaged_entries = cv.average_entries(entries)
        report.add_entries_to_report(entries)
        print "Next Step: estimate remaining metrics using validation set."
        X_train, y_train = manager.load_clean_datasets('train',project_settings)['train']
        X_train_p, y_train_p = manager.fit_transform(X_train,y_train,'train')
        model.fit(X_train_p, y_train_p)
        #if model.gen_output_flag == True:
        #    model.gen_output()
        print 'Model Fit. Next Step: Perform Model Evaluation'
        X_val, y_val = manager.load_clean_datasets('val',project_settings)['val']
        X_val_p, y_val_p = manager.transform(X_val,y_val, 'val')
        y_pred = model.predict(X_val_p)
        if model.gen_output_flag:
            model.set_features(manager.working_features)
            model.gen_output()
        evaluation_battery = load_evaluation_battery(project_settings)
        non_cv_battery = {k: v for k, v in evaluation_battery.items() if evaluation_battery[k]['metric_type'] != 'column'}
        report_row = dict()
        for metric_name in non_cv_battery:
            report_row['dataset_name'] = 'validation'
            report_row['model_name'] = model_config['model_name']
            if hasattr(cv,'optimal_hyperparams'):
                report_row['setting'] = str(cv.optimal_hyperparams)
            else:
                report_row['setting'] = 'default'
            report_row['metric'] = metric_name
            metric_class = non_cv_battery[metric_name]['class']
            kwargs = non_cv_battery[metric_name]['kwargs']
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
            y_pred = model.predict(X_test_p)
            for metric_name in non_cv_battery:
                report_row['dataset_name'] = 'test'
                report_row['model_name'] = model_config['model_name']
                if hasattr(cv, 'optimal_hyperparams'):
                    report_row['setting'] = str(cv.optimal_hyperparams)
                else:
                    report_row['setting'] = 'default'
                report_row['metric'] = metric_name
                metric_class = non_cv_battery[metric_name]['class']
                kwargs = non_cv_battery[metric_name]['kwargs']
                try:
                    content = metric_class(y_pred, y_val, **kwargs)
                except ValueError:
                    content = np.nan
                report_row['content'] = content
                report.add_entries_to_report(report_row)
        i += 1
    print "All Models Fit and Evaluated. Writing Report"
    report.write_report()
    print "\nReport written to: " + project_settings['repo_loc'] +'/' + project_settings['project_name'] + '/evaluation/report.html'

if __name__ == "__main__":
    main()