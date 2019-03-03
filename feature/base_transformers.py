import pandas as pd
import numpy as np
from numpy import random as npr
from scipy.interpolate import interp1d
from algorithms.algoutils import get_algo_class
from utils import flip_dict


class InvOneHotEncoder:

    def __init__(self,touch_indices, prior_features,val_map=None):
        self.prior_features = prior_features
        self.touch_indices = touch_indices

        if val_map is not None:
            self.val_map = val_map

        idx_val_map = dict()

        for ti in touch_indices:
            prior_feature_col = prior_features[ti]
            value = prior_feature_col.split('_', )[1]
            idx_val_map[ti] = value
            if val_map is not None:
                try:
                    assert value in val_map.keys()
                except AssertionError:
                    print value + " is not in val_map for feature " + prior_feature_col.split('_', )[0]
                    raise Exception
                idx_val_map[ti] = float(val_map[value])
            else:
                try:
                    idx_val_map[ti] = float(idx_val_map[ti])
                except ValueError:
                    print "Specify a val_map for feature " + prior_feature_col.split('_', )[0]
                    raise Exception
                    #TODO: include this in Exception message

        self.idx_val_map = idx_val_map

    def fit(self,X,y):
        pass

    def transform(self,X_touch,dataset_name):
        #Pandas dependent
        max_col = X_touch.idxmax(1)
        idx_val_map = self.idx_val_map
        num_column = max_col.apply(lambda x: idx_val_map[x])
        if hasattr(self,'val_map'):
            val_map = self.val_map
            inv_val_map = flip_dict(val_map)
            for val in inv_val_map:
                try:
                    assert val in num_column.values
                except AssertionError:
                    print '\t' + str(val) + " not in column for dataset:" + dataset_name + ". Possibly Tweak val_map in as_numeric transform"
        return pd.DataFrame(num_column,index=X_touch.index)

class Interpolator(interp1d):

    def __init__(self,x,y,**kwargs):
        super(Interpolator,self).__init__(x,y,fill_value='extrapolate',**kwargs)

    def transform(self,x):
        return self(x)

class LeaveOneOutEncoder:

    def __init__(self):
        pass

    def fit(self, X_touch, y_touch):
        loo_vals = dict()
        mean_val_dict = dict()
        y_ser = pd.Series(y_touch,index=X_touch.index)
        for j in X_touch.columns:
            j_idx = X_touch[X_touch.loc[:, j] > 0].index  #take all rows that belong to category j
            yj_dict = dict(y_ser.loc[j_idx])
            cat_sum = sum(yj_dict.values())
            cat_len = len(yj_dict)
            if cat_len > 0:
                mean_val_dict[j] = cat_sum / cat_len
            else:
                mean_val_dict[j] = y_ser.mean()
            for i in j_idx:
                if cat_len > 1:
                    y_loo_catmean = (cat_sum - yj_dict[i]) / (cat_len - 1)    #for each row in j_idx, calc mean w/o row value
                else:
                    y_loo_catmean = (y_ser.sum() - yj_dict[i]) / (len(y_ser) - 1)
                loo_vals[i] = y_loo_catmean
        self.loo_means = pd.Series(loo_vals)
        self.mean_val_dict = mean_val_dict

    def transform(self, X_touch, dataset_name):
        if dataset_name == 'train':
            loo_means = self.loo_means
            assert len(loo_means) == len(X_touch)
            assert not pd.isnull(loo_means).any()
            return pd.DataFrame(self.loo_means)
        else:
            assert len(self.mean_val_dict) == X_touch.shape[1]
            mean_val_dict = self.mean_val_dict
            cat_means = X_touch.idxmax(1).apply(lambda x: mean_val_dict[x])
            assert not pd.isnull(cat_means).any()
            return pd.DataFrame(cat_means,index=X_touch.index)

class MetaModeler:

    def __init__(self,base_algorithm, keyword_arg_settings):
        self.base_algorithm = base_algorithm
        self.keyword_arg_settings = keyword_arg_settings
        self.fitted_base_algo = None


    def fit(self,X_touch,y_touch):
        base_algorithm_name = self.base_algorithm
        keyword_arg_settings = self.keyword_arg_settings
        base_algo_class = get_algo_class(base_algorithm_name)
        base_algo_instance = base_algo_class(**keyword_arg_settings)
        base_algo_instance.fit(X_touch,y_touch)
        self.fitted_base_algo = base_algo_instance

class OOSPredictorEns:

    def __init__(self, ens_algos, ens_algos_keyword_arg_settings_dict, validation_peeking):
        self.ens_algos = ens_algos
        self.ens_algos_keyword_arg_settings_dict = ens_algos_keyword_arg_settings_dict
        self.folds_info = None
        self.oos_fitted_ensemble = None
        self.allow_peeking = validation_peeking

    def fit(self,X_touch, y_touch):
        oos_fitted_ensemble = dict()
        folds_info = self.folds_info
        assert type(folds_info) != None
        current_fold = folds_info['fold_i']
        folds_map = folds_info['folds_map']
        num_folds = len(folds_map)
        if not self.allow_peeking:
            other_folds = filter(lambda x: x != current_fold, range(num_folds))
            excluded_indices = folds_map[current_fold][1]
        else:
            other_folds = range(num_folds)
            excluded_indices = list()
        folds_map_copy = dict(zip(folds_map.keys(),[list(v) for v in folds_map.copy().values()]))
        for fold_i in other_folds:
            oos_fitted_ensemble[fold_i] = dict()
            folds_map_copy = self.filter_excluded_dev_indices(folds_map_copy, fold_i, excluded_indices)
            if len(excluded_indices) > 0:
                assert len(folds_map_copy[fold_i][0]) < len(folds_map[fold_i][0])
            ind_dev = folds_map_copy[fold_i][0]
            #existing b/c not all indices in folds guaranteed. Other HorizontalTransformers can eliminate indices
            ind_dev_existing = list(set(ind_dev).intersection(X_touch.index))
            X_f = X_touch.loc[ind_dev_existing,:]
            y_f = pd.Series(y_touch,index=X_touch.index).loc[ind_dev_existing]
            inner_excluded_indices = folds_map_copy[fold_i][1]
            assert not pd.Series([i in X_f.index for i in inner_excluded_indices]).all()
            ens_algos = self.ens_algos
            ens_algos_keyword_arg_settings_dict = self.ens_algos_keyword_arg_settings_dict
            for algo_name in ens_algos:
                algo_keyword_arg_settings = ens_algos_keyword_arg_settings_dict[algo_name]
                algo_class = get_algo_class(algo_name)
                algo_instance = algo_class(**algo_keyword_arg_settings)
                algo_instance.fit(X_f,y_f)
                oos_fitted_ensemble[fold_i][algo_name] = algo_instance
        self.oos_fitted_ensemble = oos_fitted_ensemble

    def transform(self, X_touch, y_touch, dataset_name):
        assert type(self.oos_fitted_ensemble) != None
        oos_fitted_ensemble = self.oos_fitted_ensemble
        ens_algos = self.ens_algos
        working_folds = oos_fitted_ensemble.keys()
        if dataset_name == 'train':
            folds_info = self.folds_info
            folds_map = folds_info['folds_map']
            ens_algo_cols = dict()
            for algo_name in ens_algos:
                ens_algo_cols[algo_name] = pd.Series()
                len_col = len(ens_algo_cols[algo_name])
                existing_target_idxs = list()
                for fold_i in working_folds:
                    target_idx = folds_map[fold_i][1]
                    target_idx_existing = list(set(target_idx).intersection(X_touch.index))
                    X_f = X_touch.loc[target_idx_existing,:]
                    target_algo_instance = oos_fitted_ensemble[fold_i][algo_name]
                    y_hat_vals = target_algo_instance.predict(X_f)
                    y_hat_col_target_idx = pd.Series(y_hat_vals,index=target_idx_existing)
                    ens_algo_cols[algo_name] = ens_algo_cols[algo_name].append(y_hat_col_target_idx)
                    assert len(ens_algo_cols[algo_name]) == len_col + len(target_idx_existing)
                    len_col = len(ens_algo_cols[algo_name])
                    existing_target_idxs = existing_target_idxs + target_idx_existing
        else:
            #TODO: Check this logic for leakage and fix if needed. Now it's method seen in Sverigne's NB on Kaggle
            ens_algo_cols = dict()
            for algo_name in ens_algos:
                all_fitted_algo_name_algos = [oos_fitted_ensemble[fold_i][algo_name] for fold_i in working_folds]
                mean_fitted_values = np.column_stack([model.predict(X_touch) for model in all_fitted_algo_name_algos]).mean(axis=1)
                assert len(mean_fitted_values) == len(X_touch)
                ens_algo_cols[algo_name] = mean_fitted_values
            existing_target_idxs = X_touch.index
        X_touched = pd.DataFrame(ens_algo_cols,index=existing_target_idxs)
        if self.allow_peeking:
            assert len(X_touched) == len(X_touch)
        else:
            pass
        num_cols = X_touched.shape[1]
        assert num_cols == len(ens_algos)
        X_touched.columns = range(num_cols)
        if not self.allow_peeking:
            y_touched = pd.Series(y_touch,index=X_touch.index).loc[existing_target_idxs].tolist()
        else:
            y_touched = y_touch
        return X_touched, y_touched

    def filter_excluded_dev_indices(self, folds_map_copy, fold_i, excluded_indices):
        fold_dev_ind = folds_map_copy[fold_i][0]
        folds_map_copy[fold_i][0] = filter(lambda x: x not in excluded_indices, fold_dev_ind)
        return folds_map_copy

    def set_folds_info(self,folds_info):
        self.folds_info = folds_info

class Stacker:

    def __init__(self, stacker_algo_name, keyword_arg_settings):
        self.stacker_algo_name = stacker_algo_name
        self.keyword_arg_settings = keyword_arg_settings

    def fit(self, X_touch, y_touch):
        loo_vals = dict()
        assert len(X_touch) == len(y_touch)
        y_ser = pd.Series(y_touch, index=X_touch.index)
        stacker_algo_name = self.stacker_algo_name
        #print "\t\tPerforming LOO Stacking Procedure . . . "
        cnt = 0
        for i in X_touch.index:
            X_no_i = X_touch.drop([i])
            y_no_i = y_ser.drop(i)
            stacker_algo_class = get_algo_class(stacker_algo_name)
            stacker_algo_instance = stacker_algo_class(**self.keyword_arg_settings)
            stacker_algo_instance.fit(X_no_i,y_no_i)
            reshaped_i = X_touch.loc[i].values.reshape(1,-1)
            y_hat = stacker_algo_instance.predict(reshaped_i)
            assert len(y_hat) == 1
            loo_vals[i] = y_hat[0]
            if (cnt % 100 == 0) and (cnt != 0):
                print '\t\tLOO proc completed ' + str(cnt) + ' times'
            cnt += 1
        self.loo_vals = loo_vals
        stacker_algo_full = stacker_algo_class(**self.keyword_arg_settings)
        stacker_algo_full.fit(X_touch,y_ser)
        self.stacker_algo_full = stacker_algo_full

    def transform(self, X_touch, dataset_name):
        if dataset_name == 'train':
            loo_vals = self.loo_vals
            assert (loo_vals.keys() == X_touch.index).all()
            assert not pd.isnull(loo_vals.values()).any()
            return pd.DataFrame(loo_vals.values(),index=loo_vals.keys())
            #return loo_vals.values()
        else:
            stacker_algo_full = self.stacker_algo_full
            y_hat = stacker_algo_full.predict(X_touch)
            return y_hat

class Deleter:

    def __init__(self):
        pass

    def fit(self,X_touch,y_touch):
        pass

    def transform(self,X_touch,y_touch=None):
        if y_touch is None:
            return pd.DataFrame(index=X_touch.index)
        else:
            return pd.DataFrame(), list()

class Imputer:

    def __init__(self, replace_with):
        self.replace_with = replace_with

    def fit(self, X_col, y): #TODO: Make this only an x acceptor, like Truncator above
        pass

    def transform(self, X_touch):
        replace_with = self.replace_with
        if replace_with == 'zeros':
            X_touched = X_touch.iloc[:,0].where(pd.notnull(X_touch.iloc[:,0]),other=0)
        else:
            raise Exception
        return X_touched

class Recoder:

    def __init__(self, val_map):
        self.val_map = val_map

    def fit(self, X):
        pass

    def transform(self,X_touch):
        val_map = self.val_map
        X_touched = X_touch
        X_touched.columns = range(X_touched.shape[1])
        for origin, target in val_map.items():
            X_touched.iloc[:,target] = X_touched.iloc[:,origin].where( X_touched.iloc[:,origin] != X_touched.iloc[:,target],
                                                                     other = X_touched.iloc[:,origin])
            X_touched.fillna(1,inplace=True)
            if target != origin:
                X_touched = X_touched.drop([origin],axis=1)
            else:
                pass
        return X_touched

class Identity:

    def __init__(self):
        pass

    def fit(self,X_touch,y_touch):
        pass

    def transform(self, X_mat):
        return X_mat

class Sampler:

    def __init__(self,upsample=False):
        self.touch_indices = None
        self.untouched_indices = None
        self.upsample_flag = upsample

    def transform(self,X_touch,y_touch):
        touch_indices = self.touch_indices
        untouched_indices = self.untouched_indices
        sample_size = len(untouched_indices)
        sampled_idx = npr.choice(touch_indices,size=sample_size)
        X_touch_dict = X_touch.to_dict(orient='index')
        new_x_rows = [X_touch_dict[ri] for ri in sampled_idx]
        X_touched = pd.DataFrame(new_x_rows,index=sampled_idx)
        y_touch_dict = dict(pd.Series(y_touch,index=touch_indices))
        y_touched = [y_touch_dict[ri] for ri in sampled_idx]
        return X_touched, y_touched