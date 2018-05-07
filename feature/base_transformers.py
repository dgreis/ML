import pandas as pd
from numpy import random as npr
from scipy.interpolate import interp1d

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

class Deleter(object):

    def __init__(self):
        super(Deleter, self).__init__()

    def fit(self,X_touch,y_touch):
        pass

    def transform(self,X_touch):
        return pd.DataFrame(index=X_touch.index)

class Truncator:

    def __init__(self):
        pass

    def fit(self,X_col):
        #implementing 1.5xIQR rule for now
        desc = X_col.describe()
        IQR = (desc.loc['75%'] - desc.loc['25%']).values[0]
        assert not pd.isnull(IQR)
        uthrsh = desc.loc['75%'].values[0] + 1.5*IQR
        lthrsh = desc.loc['25%'].values[0] - 1.5*IQR
        self.uthrsh = uthrsh
        self.lthrsh= lthrsh

    def transform(self,X_touch, y_touch):
        return pd.DataFrame(), list()

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