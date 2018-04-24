from __future__ import print_function

import pandas as pd
import numpy as np

from evaluation import load_evaluation_battery
from itertools import izip, product
from sklearn.model_selection import KFold
from feature.manager import Manager

class CrossValidator:

    def __init__(self, model_config, project_settings):
        if model_config['hyperparam_tuning_settings'] is not None:
            self.tune_hyperparams = True
            hyper_params = model_config['hyperparam_tuning_settings']['hyperparams']
            grid_template = dict()
            for param in hyper_params:
                grid_template[param] = eval(hyper_params[param])
                #TODO: This logic needs to be changed for new strategies, like random
            self.grid = list(dict(izip(grid_template, x)) for x in product(*grid_template.itervalues()))
            self.hyperparam_cv_folds = model_config['hyperparam_tuning_settings']['num_folds']
            self.hyperparam_eval_metric = model_config['hyperparam_tuning_settings']['eval_metric']
        else:
            self.tune_hyperparams = False
        self.evaluation_battery = load_evaluation_battery(project_settings)
        self.report_entries = dict()
        self.project_settings = project_settings
        self.model_config = model_config

    def set_optimal_hyperparams(self,model):
        assert hasattr(self,'optimal_hyperparams')
        optimal_hyperparams = self.optimal_hyperparams
        for param in optimal_hyperparams:
            setattr(model,param,optimal_hyperparams[param])
        return model

    def tune_hyperparams_via_cv(self, data, model):
        """Example in models.yaml file:
        Models:
            Model Name:
                hyperparam_tuning_settings:
                    hyperparams:
                        <param>: "[<va1l>, <val2>]"
                    num_folds: <folds>
                    eval_metric: <metric>
        """
        grid = self.grid
        num_folds = self.hyperparam_cv_folds
        i = 0
        num_settings = len(grid)
        cv_results = dict()
        results_df = pd.DataFrame({'setting': [], 'mean': [], 'sd': []})
        for setting in grid:
            for param in setting:
                setattr(model,param,setting[param])
            model_name = str(setting)
            report_entries = self.perform_cross_validation(data, model, num_folds, model_name)
            i = i + num_folds
            print("\tCV complete (" + str(i) + "/" + str(num_folds*num_settings) + ") models")
            hyperparam_eval_metric = self.hyperparam_eval_metric
            scores = report_entries[hyperparam_eval_metric][str(setting)]
            cv_results[str(setting)] = scores
            results_df = results_df.append({'setting' : str(setting)
                              ,'mean': np.mean(cv_results[str(setting)])
                              ,'sd': np.std(cv_results[str(setting)])
                               },ignore_index=True)
        means = results_df['mean']
        stds = results_df['sd']
        print("\t********* CV Results *********")
        for mean, std, params in zip(means, stds, results_df['setting']):
            print("\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        results_df = results_df.sort_values('mean',ascending=False).reset_index(drop=True)
        print("\tBest hyperparam setting: " + results_df.loc[0,'setting'] + " @ " + str(results_df.loc[0,'mean']))
        self.optimal_hyperparams = eval(results_df.loc[0,'setting'])

    def perform_cross_validation(self, data, model, num_folds, model_name):
        X_train_val, y_train_val = data['train_val']
        kf = KFold(num_folds)
        folds = list(kf.split(X_train_val,y_train_val))
        report_entries = dict()
        model_config = self.model_config
        project_settings = self.project_settings
        chef = Manager(model_config, project_settings)
        f = 1
        for fold in folds:
            ind_dev, ind_val = fold[0], fold[1]
            X_train,y_train = X_train_val.loc[ind_dev,:], pd.Series(y_train_val).loc[ind_dev].tolist()
            X_train_p, y_train_p = chef.fit_transform(X_train,y_train,'train')
            X_val,y_val = X_train_val.loc[ind_val,:], pd.Series(y_train_val).loc[ind_val].tolist()
            X_val_p, y_val_p = chef.transform(X_val, y_val, 'val')
            assert X_train_p.shape[1] == X_val_p.shape[1]
            assert len(y_val_p) == len(y_val)
            if f == 1:
                print("\tCV-fold (" + str(f) + "/" + str(num_folds) + ") data finalized. Training with " + str(len(X_train_p)) + " samples. Validation data with " +\
                  str(len(X_val_p)) + " samples. Model with " + str(X_val_p.shape[1]) + " features. Now fitting model... ",end="")
            else:
                print("\tCV-fold (" + str(f) +"/"+str(num_folds) + ") data finalized. Now fitting model... ",end="")
            model.fit(X_train_p,y_train_p)
            print("Model Fit.")
            y_pred = model.predict(X_val_p)
            evaluation_battery = self.evaluation_battery
            cv_battery = {k: v for k, v in evaluation_battery.items() if evaluation_battery[k]['metric_type'] == 'column'}
            for metric_name in cv_battery:
                metric_class = evaluation_battery[metric_name]['class']
                kwargs = evaluation_battery[metric_name]['kwargs']
                if not report_entries.has_key(metric_name):
                    report_entries[metric_name] = {model_name : []}
                report_entries[metric_name][model_name].append(metric_class(y_pred, y_val_p, **kwargs))
            f += 1
        return report_entries

    def average_entries(self,entries):
        averaged_dict = dict()
        metric_keys = entries.keys()
        for key in metric_keys:
            inner_dict = entries[key]
            model_name = inner_dict.keys()[0]
            series = inner_dict[model_name]
            averaged_dict[key] = { model_name : np.mean(series) }
        return averaged_dict




