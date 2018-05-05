from __future__ import division

import importlib
import itertools

import pandas as pd
import numpy as np
import numpy.random as npr

from manipulator import ManipulatorChain, Manipulator
from algorithms.wrapper import Wrapper
from utils import flip_dict, load_inv_column_map, load_clean_input_file_filepath
from algorithms.algoutils import get_algo_class
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

from django.utils.text import slugify

class TransformChain(ManipulatorChain):

    def __init__(self,starting_transformations, model_config, project_settings,original_columns=False):
        updated_transformations = list()
        model_config['feature_settings']['order'] = 0
        engineering_module = importlib.import_module('feature.engineering')
        for transformation in starting_transformations:
            engineering_module = importlib.import_module('feature.engineering')
            transformer_name = transformation.keys()[0].split('.')[-1:][0]
            transform_class = getattr(engineering_module, transformer_name)
            if transform_class.__bases__[0] == getattr(engineering_module,'TransformChain'):
                transform_chain_class = transform_class
                working_transformations = model_config['feature_settings']['feature_engineering']
                tc = transform_chain_class(working_transformations,model_config,project_settings,original_columns)
                updated_transformations = tc.transformations
            else:
                transformer_name = transformation.keys()[0]
                transformer_class_name = transformer_name.split('.')[-1:][0]
                transform_class = getattr(engineering_module, transformer_class_name)
                transformer = transform_class(model_config, project_settings)
                model_config['feature_settings']['order'] += 1
                transformation[transformer_name]['initialized_transformer'] = transformer
                updated_transformations = updated_transformations + [transformation]
        super(TransformChain,self).__init__(updated_transformations, model_config, project_settings,original_columns)
        self.transformations = updated_transformations

    # def fit(self,X_mat,y,dataset_name):
    #     log_prefix = "[" + dataset_name + "]"
    #     transformations = self.transformations
    #     i = 1
    #     for d in transformations:
    #         transformer_name = d.keys()[0]
    #         #print "\t" + log_prefix + " Performing feature engineering (" + str(i) + '/' + str(
    #         #    len(transformations)) + "): " + transformer_name
    #         transformer = d[transformer_name]['initialized_transformer']
    #         X_touch, X_untouched, y_touch, y_untouched = transformer.split(X_mat, y)
    #         transformer.fit(X_touch,y_touch)
    #         i += 1

    def transform(self,X_mat,y,dataset_name,fit_transform=False):
        X_mat_start = X_mat.copy()
        transformations = self.transformations
        if len(transformations) < 1:
            X_transform, y_transform = X_mat, y
        else:
            i = 1
            for d in transformations:
                transformer_name = d.keys()[0]
                transformer = d[transformer_name]['initialized_transformer']
                X_touch, X_untouched, y_touch, y_untouched = transformer.split(X_mat, y)
                if fit_transform:
                    transformer.fit(X_touch, y_touch)
                transformer_class = transformer.__class__
                transform_args = self._get_args(transformer_class, 'transform')
                additional_args = filter(lambda x: x not in ['X_touch','y_touch'], transform_args)
                kwargs = dict()
                for arg in additional_args:
                   kwargs[arg] = eval(arg)
                X_touched, y_touched = transformer.transform(X_touch, y_touch,**kwargs)
                X_transform, y_transform = transformer.combine(X_touched, X_untouched, y_touched, y_untouched)
                assert True not in pd.isnull(X_transform).any(1).value_counts() #TODO: pandas dependent
                if dataset_name == "train":
                    if transformer.store:
                        artifact_dir = self.artifact_dir
                        transformer.store_output(X_transform,output_dir=artifact_dir)
                X_mat, y_mat = X_transform, y_transform
                i += 1
        return X_transform, y_transform

    def fit_transform(self,X_mat,y,dataset_name):
        return self.transform(X_mat,y,dataset_name,fit_transform=True)

class Transformer(Manipulator):

    def __init__(self, model_config, project_settings):
        manipulations = model_config['feature_settings']['feature_engineering']
        super(Transformer,self).__init__(model_config,project_settings,manipulations)
        transformer_name = self.manipulator_name
        transformer_settings = self.fetch_transform_settings(model_config,transformer_name)
        self.base_transformer = None
        if transformer_settings.has_key('kwargs'):
            kwargs = transformer_settings['kwargs']
        else:
            kwargs = dict()
        self.kwargs = kwargs
        if transformer_settings.has_key('store_output'):
            self.store = True
        else:
            self.store = False
        self.inclusion_patterns = transformer_settings['inclusion_patterns']

    def det_prior_feature_names_filepath(self,model_config):
        project_settings = self.project_settings
        if not model_config['feature_settings']['select_before_eng']:
            prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            filters = model_config['feature_settings']['feature_selection']
            num_filters = len(filters)
            if num_filters > 0:
                last_filter_name = filters[-1:][0].keys()[0]
                prior_manipulator_feature_names_filepath = self._det_output_features_filepath(last_filter_name)
            else:
                prior_manipulator_feature_names_filepath = load_clean_input_file_filepath(project_settings,'feature_names')
        return prior_manipulator_feature_names_filepath

    def configure_ancestors_and_features(self):
        prior_transform_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        prior_inv_col_map = load_inv_column_map(prior_transform_feature_names_filepath)
        prior_features = flip_dict(prior_inv_col_map)
        self.prior_features = prior_features
        touch_indices, untouched_indices = self.determine_split_indices(prior_features)
        new_features = self.gen_new_column_names(touch_indices, prior_features)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        new_feature_set = self.reindex(prior_features, new_features)
        self.features = new_feature_set
        self.output_features()

    def fetch_transform_settings(self,model_config, transformer_name):
        feature_eng_settings = model_config['feature_settings']['feature_engineering']
        for item in feature_eng_settings:
            if item.keys()[0] == transformer_name:
                transform_settings = item[transformer_name]
            else:
                pass
        return transform_settings

    def set_base_transformer(self,transformer_instance):
        self.base_transformer = transformer_instance

    def fit(self,X_touch,y_touch,**kwargs):
        self.base_transformer.fit(X_touch, y_touch)

    def determine_split_indices(self, prior_features):
        #TODO: Better name this fn?
        assert getattr(self, 'inclusion_patterns') != None
        include_columns = list()
        col_indices = prior_features.keys()
        col_names = prior_features.values()
        inclusion_patterns = self.inclusion_patterns
        inv_working_features = flip_dict(prior_features)
        if inclusion_patterns == ['All']:
            touch_indices = range(len(prior_features))
            untouched_indices = list()
            exclusion_flag = False
        else:
            if type(inclusion_patterns) == dict:
                assert inclusion_patterns.keys() == ['All But']
                exclusion_flag = True
                inclusion_patterns = inclusion_patterns['All But']
            else:
                exclusion_flag = False
            for pattern in inclusion_patterns:
                len_pat = len(pattern)
                pattern_begin_cols = filter(lambda x: x[0:len_pat] == pattern, col_names)
                include_columns = include_columns + pattern_begin_cols
            non_inter_include_columns = filter(lambda x: 'x%x' not in x, include_columns)
            touch_indices = [int(inv_working_features[col_name]) for col_name in non_inter_include_columns]
            untouched_indices = list(set(col_indices).difference(set(touch_indices)))
            touch_indices = touch_indices
            untouched_indices = untouched_indices
        if not exclusion_flag:
            return touch_indices, untouched_indices
        else:
            return untouched_indices, touch_indices

    def transform(self, X_touch, y_touch, **kwargs):
        X_touched = self.base_transformer.transform(X_touch,**kwargs)
        if type(X_touched) != pd.core.frame.DataFrame: #TODO: fix this when I move out of pandas
            rdf = pd.DataFrame(X_touched,index=X_touch.index)
        else:
            rdf = X_touched
        return rdf, y_touch

    def combine(self,X_touched,X_untouched,y_touched,y_untouched):
        """Pandas Dependent"""
        if X_untouched.shape[1] > 0:
            X_transform = pd.merge(X_untouched, X_touched, left_index=True, right_index=True)
            assert len(X_transform) == len(X_untouched) == len(X_touched)
        else:
            X_transform = X_touched
        features = self.features
        assert len(features) == X_transform.shape[1]
        X_transform.columns = range(len(features))
        if y_untouched is not None:
            print "this is mean to be a vertical transform. y_touched is not None which seems like a horizontal transform"
            raise Exception
        return X_transform, y_touched

    def store_output(self,X_mat,output_dir): #TODO: When I need to output filter, make this an abstract method in new ManipulatorChain class
        model_name = self.model_name
        transform_name = str(self.__class__).split('.')[-1:][0]
        output_file_name = slugify(model_name + '-' + transform_name)
        X_mat.to_csv(output_dir + '/' + output_file_name + '.txt', header=False, index=False, sep='\t')  # TODO: fix this when I move out of pandas


class basis_expansion(Transformer):
    """
    yaml usage example:

    Models:
      <Model Name>:
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - basis_expansion:
                 inclusion_patterns:
                  - <feature_name>
                 kwargs:
                    include_bias: <bool>
                    interaction_only: <bool>
    """
    def __init__(self, model_config, project_settings):
        super(basis_expansion, self).__init__(model_config, project_settings)
        self.set_base_transformer(PolynomialFeatures(**self.kwargs))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        tuples = list()
        for i in range(len(orig_tcol_idx)):
            b = orig_tcol_idx[i]
            rem_list = orig_tcol_idx[i:]
            t = [(working_features[b],working_features[r_i]) for r_i in rem_list]
            tuples = tuples + t
        stiched_feature_names = [self.stich(x) for x in tuples]
        orig_tcol_names = [working_features[idx] for idx in orig_tcol_idx]
        Xt_feat_names = orig_tcol_names + stiched_feature_names
        if getattr(self.base_transformer,'include_bias'):
            Xt_feat_names = ['bias'] + Xt_feat_names
        if getattr(self.base_transformer,'interaction_only'):
            Xt_feat_names = filter(lambda x: '**' not in x, Xt_feat_names)
        return Xt_feat_names


    def stich(self,tuple):
        if tuple[0] == tuple[1]:
            return "(" + tuple[0] + ')**2'
        else:
            return 'inter(' + tuple[0] + 'x%x' + tuple[1] + ')'

class interaction_terms(TransformChain):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - interaction_terms:
                  interactions: (examples below)
                    - "('bill_sep','hist_sep')"
                    - "('bill_sep','prepay_sep')"
    """

    def __init__(self,transformations, model_config, project_settings,original_columns):
        Manipulator.__init__(self,model_config,project_settings,transformations) #TODO: figure out this troublesome line
        raw_interaction_strs = filter(lambda x: x.keys()[0] == 'interaction_terms',  transformations)[0]['interaction_terms']['interactions']
        compact_interactions = [eval(ris) for ris in raw_interaction_strs]
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_idx = transformer_names.index('interaction_terms')
        if t_idx == 0:
            if model_config['feature_settings']['select_before_eng']:
                print "Interaction terms are meant to be run before model selection methods. Turn off select_before_eng flag"
                raise Exception
            prior_transform_feature_names_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        else:
            prior_transform = transformer_names[t_idx - 1]
            prior_transform_feature_names_filepath = self._det_output_features_filepath(prior_transform)
        inv_column_map = load_inv_column_map(prior_transform_feature_names_filepath)
        feature_names = inv_column_map.keys()
        i = 0
        for tuple in compact_interactions:
            t0 = tuple[0]
            t1 = tuple[1]
            # #TODO: Make this check work for categorical variables
            # if False in [x in feature_names for x in (t0,t1)]:
            #     print "Expected columns not present in data to do interaction. Check that they have not been transformed previously"
            #     raise Exception
            #     #TODO: move exception message inside Exception to be more informative.
            len_t0_str = len(t0)
            len_t1_str = len(t1)
            t0_col_vals = filter(lambda x: x[0:len_t0_str] == t0 , feature_names)
            t1_col_vals = filter(lambda x: x[0:len_t1_str] == t1 , feature_names)
            expanded_interactions = list(itertools.product(t0_col_vals ,t1_col_vals))
            for inter in expanded_interactions:
                transformation_dict = dict()
                transformation_dict['int' + str(i) + '.' + 'basis_expansion'] = {
                    'inclusion_patterns':[inter[0], inter[1]],
                    'kwargs' : {'include_bias': False, 'interaction_only': True}
                }
                expanded_transformations.append(transformation_dict)
                i += 1
        if t_idx == 0:
            transformations = expanded_transformations + transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(transformations) - 1:
            transformations = transformations[0:t_idx] + expanded_transformations + transformations[t_idx + 1:]
            exp_idx = len(transformations[0:t_idx]) + len(expanded_transformations)
        else:
            transformations = transformations[:-1] + expanded_transformations
            exp_idx = len(transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = transformations
        super(interaction_terms, self).__init__(transformations[:exp_idx], model_config, project_settings,original_columns)

class normalize(Transformer):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - normalize:
                  inclusion_patterns:
                    - <pattern>
     (You can also specify 'All Numeric' to normalize all numeric columns, provided
     that you include a list of the numeric feature names in the project_settings yaml
     file in the src folder of the project directory)
    """
    def __init__(self, model_config, project_settings):
        super(normalize, self).__init__(model_config, project_settings )
        self.set_base_transformer(Normalizer(**self.kwargs))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'norm(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names


class standard_scale(Transformer):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            feature_engineering:
              - standard_scale:
                  inclusion_patterns:
                    - <pattern>
     (You can also specify 'All Numeric' to scale all numeric columns, provided
     that you include a list of the numeric feature names in the project_settings yaml
     file in the src folder of the project directory)
    """
    def __init__(self, model_config, project_settings):
        super(standard_scale, self).__init__(model_config, project_settings )
        self.set_base_transformer(StandardScaler(**self.kwargs))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        Xt_feat_names = list()
        for idx in orig_tcol_idx:
            base_feature_name = working_features[idx]
            norm_feature_name = 'scaled(' + base_feature_name + ')'
            Xt_feat_names.append(norm_feature_name)
        return Xt_feat_names

class pca(Transformer):
    """
    example yaml usage:
<model_name>
    feature_settings:
      feature_engineering:
        - pca:
            inclusion_patterns:
             - 'All'
            kwargs:
              n_components: <int>
    """

    def __init__(self, model_config, project_settings):
        super(pca, self).__init__(model_config, project_settings )
        self.set_base_transformer(PCA(**self.kwargs))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        num_comps = self.base_transformer.n_components
        new_col_list = list()
        for i in range(num_comps):
            new_col_name = 'pc_' + str(i)
            new_col_list.append(new_col_name)
        return new_col_list

class loo_encoding(Transformer):
    """Example in models.yaml file:
        Models:
          Model Name:
           feature_settings:
             feature_engineering:
               - loo_encoding
                   inclusion_patterns
                     - <pattern>
    """
    def __init__(self, model_config, project_settings):
        super(loo_encoding, self).__init__(model_config, project_settings )
        self.set_base_transformer(LeaveOneOutEncoder(**self.kwargs))
        self.configure_ancestors_and_features()

    # def fit_transform(self, X_touch, working_features,y):
    #     kwargs = dict()
    #     kwargs['y'] = y
    #     return super(loo_encoding, self).fit_transform(X_touch,working_features,**kwargs)

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch,y_touch, dataset_name)

    def gen_new_column_names(self, orig_tcol_idx, working_features):
        inclusion_patterns = self.inclusion_patterns
        assert len(inclusion_patterns) == 1
        pattern = inclusion_patterns[0]
        new_feature_name = 'loo(' + pattern  + ')'
        return [new_feature_name]


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

    def transform(self, X_touch, y_touch, dataset_name):
        if dataset_name == 'train':
            loo_means = self.loo_means
            assert len(loo_means) == len(X_touch)
            assert not pd.isnull(loo_means).any()
            return pd.DataFrame(self.loo_means), y_touch
        else:
            assert len(self.mean_val_dict) == X_touch.shape[1]
            mean_val_dict = self.mean_val_dict
            cat_means = X_touch.idxmax(1).apply(lambda x: mean_val_dict[x])
            assert not pd.isnull(cat_means).any()
            return pd.DataFrame(cat_means,index=X_touch.index), y_touch


class interpolate(Transformer):
    """Example in models.yaml file:
    Models:
      <model name>:
       feature_settings:
         feature_engineering:
           - interpolate
               inclusion_patterns
                 - <pattern>
                kwargs:
                    lowess: {}
                    interp1d: {}
    """
    def __init__(self, model_config, project_settings):
        super(interpolate, self).__init__(model_config, project_settings)
        self.set_base_transformer(None)
        self.configure_ancestors_and_features()

    def fit(self,X_mat,y):
        kwargs = self.kwargs
        lowess_kwargs = kwargs['lowess']
        interp1d_kwargs = kwargs['interp1d']

        assert X_mat.shape[1] == 1 #For now, just one feature per transformer

        fitted_lowess = lowess(X_mat.iloc[:,0], np.array(y),**lowess_kwargs)

        # unpack the lowess smoothed points to their values
        lowess_y = list(zip(*fitted_lowess))[0]
        lowess_x = list(zip(*fitted_lowess))[1]

        # run scipy's interpolation. There is also extrapolation I believe
        f = Interpolator(lowess_x, lowess_y,bounds_error=False,**interp1d_kwargs)

        self.set_base_transformer(f)

    def gen_new_column_names(self, touch_indices, prior_features):
        assert len(touch_indices) == 1
        touch_idx = touch_indices[0]
        old_feature_name = prior_features[touch_idx]
        new_feature_name = 'interp(' + old_feature_name + ')'
        return [new_feature_name]

class Interpolator(interp1d):

    def __init__(self,x,y,**kwargs):
        super(Interpolator,self).__init__(x,y,fill_value='extrapolate',**kwargs)

    def transform(self,x):
        return self(x)

class as_numeric(TransformChain):
    """Example in models.yaml file:
    Models:
      <model name>:
       feature_settings:
         feature_engineering:
           - as_numeric
               inclusion_patterns
                 - <pattern>
               val_maps:
                 - <pattern> : { str:val, str:val,...}
    """

    def __init__(self,transformations, model_config, project_settings,original_columns):
        Manipulator.__init__(self, model_config, project_settings,
                             transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in transformations]
        t_idx = transformer_names.index('as_numeric')
        i = 0
        as_numeric_entry = filter(lambda x: x.keys()[0] == 'as_numeric', transformations)[0]['as_numeric']
        inclusion_patterns = as_numeric_entry['inclusion_patterns']
        if as_numeric_entry.has_key('val_maps'):
            val_maps = as_numeric_entry['val_maps']
        else:
            val_maps = dict()
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_as_numeric'
            transformation_dict[expanded_transformer_name] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
            if val_maps.has_key(pattern):
                transformation_dict[expanded_transformer_name]['val_map'] = val_maps[pattern]
        if t_idx == 0:
            transformations = expanded_transformations + transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(transformations) - 1:
            transformations = transformations[0:t_idx] + expanded_transformations + transformations[t_idx + 1:]
            exp_idx = len(transformations[0:t_idx]) + len(expanded_transformations)
        else:
            transformations = transformations[:-1] + expanded_transformations
            exp_idx = len(transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = transformations
        super(as_numeric, self).__init__(transformations[:exp_idx], model_config, project_settings, original_columns)

class ind_as_numeric(Transformer):
    """
    This is a utility transformation, used by as_numeric transformchain transformer
    """
    def __init__(self,model_config, project_settings):
        super(ind_as_numeric, self).__init__(model_config, project_settings)
        self.configure_ancestors_and_features()
        ind_as_numeric_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
                        ['feature_engineering'])[0][self.manipulator_name]
        kwargs = dict()
        if ind_as_numeric_entry.has_key('val_map'):
            kwargs['val_map'] = ind_as_numeric_entry['val_map']
        self.set_base_transformer(InvOneHotEncoder(self.touch_indices, self.prior_features,**kwargs))


    def gen_new_column_names(self, touch_indices, prior_features):
        prior_feature_cols = [prior_features[ti] for ti in touch_indices]
        base_names = [pfc.split('_')[0] for pfc in prior_feature_cols]
        assert len(set(base_names)) == 1
        base_name = 'as_numeric(' + base_names[0] + ')'
        return [base_name]

    def transform(self, X_touch, y_touch, dataset_name):
        return self.base_transformer.transform(X_touch,y_touch,dataset_name)

class InvOneHotEncoder:

    def __init__(self,touch_indices, prior_features,val_map=None):
        self.prior_features = prior_features
        self.touch_indices = touch_indices

        if val_map is not None:
            self.val_map = val_map

        idx_val_map = dict()

        for ti in touch_indices:
            prior_feature_col = prior_features[ti]
            value = prior_feature_col.split('_')[1]
            idx_val_map[ti] = value
            if val_map is not None:
                try:
                    assert value in val_map.keys()
                except AssertionError:
                    print value + " is not in val_map for feature " + prior_feature_col.split('_')[0]
                    raise Exception
                idx_val_map[ti] = float(val_map[value])
            else:
                try:
                    idx_val_map[ti] = float(idx_val_map[ti])
                except ValueError:
                    print "Specify a val_map for feature " + prior_feature_col.split('_')[0]
                    raise Exception
                    #TODO: include this in Exception message

        self.idx_val_map = idx_val_map

    def fit(self,X,y):
        pass

    def transform(self,X_touch,y_touch,dataset_name):
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
        return pd.DataFrame(num_column,index=X_touch.index), y_touch

class sample(Transformer):
    """Example in models.yaml file:
        Models:
          <Model Name>:
           feature_settings:
             feature_engineering:
               - sample:
                   inclusion_patterns:
                     - "All"
                   kwargs:
                      upsample: <boolean> (def: False):
    """
    def __init__(self,model_config,project_settings):
        super(sample, self).__init__(model_config, project_settings )
        self.set_base_transformer(Sampler(**self.kwargs))
        self.configure_ancestors_and_features()

    def fit(self,X_mat,y_mat):
        X_mat_idx = X_mat.index.tolist()
        y = pd.Series(y_mat,index=X_mat_idx)
        min_idx = y[y == 1].index.tolist()
        maj_idx = y[y == 0].index.tolist()
        upsample = getattr(self.base_transformer,'upsample_flag')
        if upsample:
            touch_indices = min_idx
            untouched_indices = maj_idx
        else:
            touch_indices = maj_idx
            untouched_indices = min_idx
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        setattr(self.base_transformer,'touch_indices', touch_indices)
        setattr(self.base_transformer,'untouched_indices', untouched_indices)

    def split(self, X_mat, y, dataset_name):
        """This is a horizontal splitting method"""
        if dataset_name == "train":
            X_mat_idx = X_mat.index.tolist()
            untouched_indices = self.untouched_indices
            touch_indices = self.touch_indices
            X_untouched = X_mat.loc[untouched_indices,:]
            y_untouched = pd.Series(y,index=X_mat_idx).loc[untouched_indices].tolist()
            X_touch = X_mat.loc[touch_indices,:]
            y_touch = pd.Series(y,index=X_mat_idx).loc[touch_indices].tolist()
            return X_touch, X_untouched, y_touch, y_untouched
        else:
            return None, X_mat, None, y

    def transform(self,X_touch,y_touch,dataset_name):
        if dataset_name == "train":
            X_touched, y_touched = self.base_transformer.transform(X_touch, y_touch)
            return X_touched, y_touched
        else:
            return X_touch, y_touch

    def combine(self,X_touched,X_untouched,y_touched,y_untouched,dataset_name):
        "This is a vertical combine instead of the default horizontal"
        if dataset_name == "train":
            X_transform = X_untouched.append(X_touched,ignore_index=True)
            y_transform = y_touched + y_untouched
            return X_transform, y_transform
        else:
            return X_untouched, y_untouched

    def gen_new_column_names(self, touch_indices, prior_features):
        return prior_features

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

class predict(Transformer):

    def __init__(self,model_config,project_settings):
        super(predict, self).__init__(model_config, project_settings)
        predict_entry = filter(lambda x: x.keys()[0] == self.manipulator_name, model_config['feature_settings']
                        ['feature_engineering'])[0][self.manipulator_name]
        base_algoritm = predict_entry['base_algorithm']
        base_algo_class = get_algo_class(base_algoritm)
        if not predict_entry.has_key('keyword_arg_settings'):
            predict_entry['keyword_arg_settings'] = dict()
        if not predict_entry.has_key('other_options'):
            predict_entry['other_options'] = dict()
        predict_entry['model_name'] = self.manipulator_name
        self.set_base_transformer(Wrapper(base_algo_class, predict_entry,project_settings))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, touch_indices, prior_features):
        new_col_name = self.manipulator_name + '_hat'
        feature_names = [prior_features[i] for i in touch_indices]
        return feature_names + [new_col_name]

    def transform(self,X_touch,y_touch):
        base_algo_instance = self.base_transformer
        col_loc = X_touch.columns.max() + 1
        ow = X_touch.shape[1]
        X_touch.loc[:,col_loc] = base_algo_instance.predict(X_touch)
        assert X_touch.shape[1] > ow
        return X_touch, y_touch

class reset_data(Transformer):

    def __init__(self,model_config,project_settings):
        super(reset_data, self).__init__(model_config, project_settings )
        self.set_base_transformer(Resetter(**self.kwargs))
        self.configure_ancestors_and_features()

    def gen_new_column_names(self, touch_indices, prior_features):
        project_settings = self.project_settings
        clean_feature_filepath = load_clean_input_file_filepath(project_settings, 'feature_names')
        clean_features = load_inv_column_map(clean_feature_filepath)
        return clean_features.keys()

    def transform(self,X_touch,y_touch,X_mat_start):
        return self.base_transformer.transform(X_mat_start), y_touch

class Resetter:

    def __init__(self):
        pass

    def fit(self,X_touch,y_touch):
        pass

    def transform(self,X_mat_start):
        return X_mat_start





