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
            self.hyperparam_eval_metric = model_config['hyperparam_tuning_settings']['eval_metric']
        else:
            self.tune_hyperparams = False
            self.grid = ['default']
        self.cv_num_folds = project_settings['assessment']['cv_num_folds']
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

    def perform_cross_validation(self, data, model):
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
        num_settings = len(grid)
        num_folds = self.cv_num_folds
        report_entries = pd.DataFrame({'dataset_name':[],'model_name':[],'setting': [], 'metric': [],'content': []})
        i = 1
        for setting in grid:
            if grid != ['default']:
                for param in setting:
                    setattr(model.base_algorithm,param,setting[param])
            setting_name = str(setting)
            print("\ttuning hyperparam for setting: " + setting_name + " (" + str(i) +'/' + str(num_settings) + ")")
            setting_folds_entries = self.do_folds(data, model, num_folds, setting_name)
            setting_entries = self.aggregate_folds(setting_folds_entries)
            report_entries = report_entries.append(setting_entries,ignore_index=True)
            i += 1
        report_entries = self.filter_by_optimal_setting(report_entries)
        return report_entries

    def aggregate_folds(self,report_entries):
        f = {'content': ['mean', 'std']}
        g = report_entries.groupby([ 'dataset_name','model_name','setting','metric']
                               ,as_index=False).agg(f)
        new_col_names = ['_'.join(col).strip('_') for col in g.columns.values]
        g.columns = new_col_names
        g['content'] = g.apply(lambda x: (x['content_mean'],x['content_std']),axis=1)
        return g.drop(['content_mean','content_std'],axis=1)

    def filter_by_optimal_setting(self,report_entries):
        if 'default' in report_entries['setting'].values:
            return report_entries
        else:
            hyperparam_eval_metric = self.hyperparam_eval_metric
            score_df = report_entries[report_entries.loc[:,'metric'] == hyperparam_eval_metric].copy()
            means = score_df.loc[:,'content'].apply(lambda x: x[0])
            stds = score_df.loc[:,'content'].apply(lambda x: x[1])
            print("\t********* CV Results *********")
            for mean, std, params in zip(means, stds, score_df['setting']):
                print("\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            score_df.loc[:,'sort_value'] = score_df.loc[:,'content'].apply(lambda x: x[0])
            problem_type = self.project_settings['ml_problem_type'] #TODO: Fix this if it turns out this hacky way of minimizing regression and maximizing classification metrics doesn't work
            if problem_type == 'regression':
                ascending_value = True
            else:
                ascending_value = False
            score_df.sort_values('sort_value', ascending=ascending_value,inplace=True)
            score_df.reset_index(drop=True,inplace=True)
            print("\tBest hyperparam setting: " + score_df.loc[0, 'setting'] + " @ " + str(score_df.loc[0, 'sort_value']))
            self.optimal_hyperparams = eval(score_df.loc[0, 'setting'])
            winning_setting = score_df.loc[0, 'setting']
            return report_entries[report_entries['setting'] == winning_setting]

    def do_folds(self, data, model, num_folds, setting_name):
        X_train_val, y_train_val = data['train_val']
        kf = KFold(num_folds)
        folds = list(kf.split(X_train_val,y_train_val))
        report_entries = pd.DataFrame()
        model_config = self.model_config
        project_settings = self.project_settings
        manager = Manager(model_config, project_settings)
        f = 1
        for fold in folds:
            ind_dev, ind_val = fold[0], fold[1]
            X_train,y_train = X_train_val.loc[ind_dev,:], pd.Series(y_train_val).loc[ind_dev].tolist()
            X_train_p, y_train_p = manager.fit_transform(X_train,y_train,'train')
            X_val,y_val = X_train_val.loc[ind_val,:], pd.Series(y_train_val).loc[ind_val].tolist()
            X_val_p, y_val_p = manager.transform(X_val, y_val, 'val')
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
            report_row = dict()
            for metric_name in cv_battery:
                report_row['dataset_name'] = 'cross_validation'
                report_row['model_name'] = model_config['model_name']
                report_row['setting'] = setting_name
                report_row['metric'] = metric_name
                report_row['fold'] = f
                metric_class = evaluation_battery[metric_name]['class']
                kwargs = self._handle_kwargs(evaluation_battery[metric_name]['kwargs'])
                try:
                    content = metric_class(y_pred, y_val_p, **kwargs)
                except ValueError:
                    content = np.nan
                report_row['content'] = content
                report_entries = report_entries.append(report_row,ignore_index=True)
            f += 1
        return report_entries

    def _handle_kwargs(self,kwargs):
        newdict = dict()
        for k,v in kwargs.items():
            if v in ["None","True","False"]:
                newdict[k] = eval(v)
            else:
                newdict[k] = v
        return newdict




