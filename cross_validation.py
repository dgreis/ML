import pandas as pd
import numpy as np

from evaluation import load_metric_class
from itertools import izip, product
from sklearn.model_selection import KFold

class Cross_Validator:

    def __init__(self, model_config):
        hyper_params = model_config['cross_validation_settings']['hyper_params']
        grid_template = dict()
        for param in hyper_params:
            grid_template[param] = eval(hyper_params[param])
        #TODO: This logic needs to be changed for new strategies, like random
        self.grid = list(dict(izip(grid_template, x)) for x in product(*grid_template.itervalues()))

    def set_optimal_hyperparams(self,model):
        assert hasattr(self,'optimal_hyperparams')
        optimal_hyperparams = self.optimal_hyperparams
        for param in optimal_hyperparams:
            setattr(model,param,optimal_hyperparams[param])
        return model

    def perform_cross_validation(self, X_train, y_train, model, model_config):
        grid = self.grid
        num_folds = model_config['cross_validation_settings']['num_folds']
        eval_metric_name = model_config['cross_validation_settings']['eval_metric']['name']
        kf = KFold(num_folds)
        folds = list(kf.split(X_train,y_train))
        cv_results = dict()
        num_settings = len(grid)
        print "\tRunning CV..."
        i = 0
        results_df = pd.DataFrame({'setting':[],'mean':[],'sd':[]})
        for setting in grid:
            for param in setting:
                setattr(model,param,setting[param])
            scores = list()
            for fold in folds:
                ind_dev, ind_val = fold[0], fold[1]
                X_dev,y_dev = X_train.loc[ind_dev,:], pd.Series(y_train).loc[ind_dev].tolist()
                X_val,y_val = X_train.loc[ind_val,:], pd.Series(y_train).loc[ind_val].tolist()
                model.fit(X_dev,y_dev)
                print "\tCV complete for (" + str(i+1) + "/" + str(num_folds*num_settings) + ") models"
                i += 1
                y_pred = model.predict(X_val)
                metric_class = load_metric_class(eval_metric_name)
                if not model_config['cross_validation_settings']['eval_metric'].has_key('kwargs'):
                    kwargs = {}
                else:
                    kwargs = model_config['cross_validation_settings']['eval_metric']['kwargs']
                score = metric_class(y_pred, y_val,**kwargs)
                scores.append(score)
            cv_results[str(setting)] = scores
            results_df = results_df.append({'setting' : str(setting)
                              ,'mean': np.mean(cv_results[str(setting)])
                              ,'sd': np.std(cv_results[str(setting)])
                               },ignore_index=True)
        means = results_df['mean']
        stds = results_df['sd']
        print "\t********* CV Results *********"
        for mean, std, params in zip(means, stds, results_df['setting']):
            print"\t%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)
        results_df = results_df.sort_values('mean',ascending=False).reset_index(drop=True)
        print "\tBest hyperparam setting: " + results_df.loc[0,'setting'] + " @ " + str(results_df.loc[0,'mean'])
        self.optimal_hyperparams = eval(results_df.loc[0,'setting'])




