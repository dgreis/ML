import importlib
import itertools
import time
import re

import pandas as pd
import numpy as np

from feature.engineering import Cleaner, HorizontalTransformer
from feature.manipulator import ManipulatorChain, Manipulator
from utils import get_args


class TransformChain(ManipulatorChain):

    def __init__(self, transform_chain_id, starting_transformations, model_config, project_settings):
        updated_transformations = list()
        for transformer_entry in starting_transformations:
            transformer_id = list(transformer_entry.keys())[0]
            transformer_class_name = list(transformer_entry.keys())[0].split('.')[-1:][0]
            transformer_class = self._get_transformer_class(transformer_class_name)
            if issubclass(transformer_class, TransformChain):
                transform_chain_class = transformer_class
                tc = transform_chain_class(transformer_id, starting_transformations, model_config, project_settings)
                #updated_transformations = model_config['feature_settings']['manipulations']
                updated_transformations = tc.transformations
                #switched line above back to what it was before. I ran into issue where filters were being included
                #on transformchains when they didn't belong, during forest_cover project (had interactions then a filter
                #for filter selection. Keep in mind this case when this change blows something else up.
            else:
                transformer_instance = transformer_class(transformer_id, model_config, project_settings)
                transformer_entry[transformer_id]['initialized_manipulator'] = transformer_instance
                updated_transformations = updated_transformations + [transformer_entry]
        super(TransformChain, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)
        self.transformations = updated_transformations
        assert len(self.transformations) > 0

    def _get_transformer_class(self, transformer_class_name):
        engineering_module = importlib.import_module('feature.engineering')
        transformchain_module = importlib.import_module('feature.transformchain')
        tagger_module = importlib.import_module('feature.tagger')
        if hasattr(engineering_module, transformer_class_name):
            transformer_class = getattr(engineering_module, transformer_class_name)
        elif hasattr(transformchain_module, transformer_class_name):
            transformer_class = getattr(transformchain_module, transformer_class_name)
        elif hasattr(tagger_module, transformer_class_name):
            transformer_class = getattr(tagger_module, transformer_class_name)
        else:
            raise Exception
        return transformer_class

    def transform(self,X_mat,y,dataset_name,fit_transform=False):
        X_mat_start = X_mat.copy()
        transformations = self.transformations
        if len(transformations) < 1:
            X_transform, y_transform = X_mat, y
        else:
            i = 1
            for d in transformations:
                transformer_name = list(d.keys())[0]
                transformer = d[transformer_name]['initialized_manipulator']
                transformer_class = transformer.__class__
                if fit_transform:
                    le = self.leak_enforcer
                    leak_exists = le.check_for_leak(X_mat)
                    leak_allowed = le.check_leak_allowed(transformer_name)
                    if leak_exists:
                        if leak_allowed:
                            #TODO: output print statements like this to log file to have record somewhere
                            #print "\t\tLeak is allowed for " + transformer_name + ". CV Metrics will be invalid"
                            X_dev, y_dev = X_mat, y
                        else:
                            #print "\t\tLeak found for manipulator: " + transformer_name +". Removing leaked indices . . ."
                            X_dev, y_dev = le.remove_leaking_indices(X_mat, y)
                    else:
                        X_dev, y_dev = X_mat, y
                    cf_args = get_args(transformer_class, 'configure_features')
                    cf_kwargs = dict()
                    for arg in cf_args:
                        cf_kwargs[arg] = eval(arg)
                    transformer.set_inclusion_columns()
                    transformer.configure_features(**cf_kwargs)
                    touch_cols = transformer.touch_indices
                    X_rel = X_dev.loc[:,touch_cols]
                    transformer.fit(X_rel, y_dev)
                    if leak_exists and not leak_allowed:
                        if issubclass(transformer.__class__, HorizontalTransformer):
                            X_rel, y_dev = X_mat, y
                            transformer.update_touch_indices(X_mat)
                split_args = get_args(transformer_class, 'split')
                additional_args = filter(lambda x: x not in ['X_mat','y'], split_args)
                spkwargs = dict()
                for arg in additional_args:
                    spkwargs[arg] = eval(arg)
                X_touch, X_untouched, y_touch, y_untouched = transformer.split(X_mat, y,**spkwargs)
                transform_args = get_args(transformer_class, 'transform')
                additional_args = filter(lambda x: x not in ['X_touch','y_touch'], transform_args)
                tfkwargs = dict()
                for arg in additional_args:
                   tfkwargs[arg] = eval(arg)
                #TODO: Write a less confusing routing logic for transformers
                X_touched, y_touched = transformer.transform(X_touch, y_touch,**tfkwargs)
                combine_args = get_args(transformer_class, 'combine')
                additional_args = filter(lambda x: x not in ['X_touched','X_untouched','y_touched','y_untouched'],combine_args)
                cmkwargs = dict()
                for arg in additional_args:
                    cmkwargs[arg] = eval(arg)
                X_transform, y_transform = transformer.combine(X_touched, X_untouched, y_touched, y_untouched,**cmkwargs) #TODO: clean up this kwarg loading mess
                if dataset_name == "train":
                    if transformer.store:
                        artifact_dir = self.artifact_dir
                        transformer.store_output(X_transform,output_dir=artifact_dir)
                X_mat, y = X_transform, y_transform
                i += 1
            try:
                self._assert_no_nulls(X_transform)
            except AssertionError:
                features = transformer.features
                nis = pd.isnull(X_transform).any(0)[pd.isnull(X_transform).any(0)].index
                still_null_cols = [features[ni] for ni in nis]
                still_null_str = ('\n\t\t').join(still_null_cols)
                print("\tIf you specified a handle_missing_data transformer, your strategy didn't succeed in handling " \
                      "all the missing data. Still missing data for following cols:\n\t\t" +
                       still_null_str +"\nConsider adding a catch-all 'delete_obs as your final sub_transformer." + \
                       "or adding additional strategies to handle variables above.")
                time.sleep(0.001)
                #TODO: Figure out more helpful Exceptions. Till then, time.sleep above makes it a bit more readable
                raise Exception
        return X_transform, y_transform

    def _assert_no_nulls(self,X_transform):
        if type(X_transform) == pd.DataFrame:
            try:
                assert True not in pd.isnull(X_transform).any(1).value_counts()
                return True
            except AssertionError:
                return False
        elif type(X_transform) == np.ndarray:
            try:
                assert True not in pd.isnull(X_transform).any(1)
                return True
            except AssertionError:
                return False
        else:
            raise Exception("Data is not in recognized format")


    def fit_transform(self,X_mat,y,dataset_name):
        return self.transform(X_mat,y,dataset_name,fit_transform=True)

    def update_manipulations_and_transformations(self, expanded_transformations):
        model_config = self.model_config
        manipulator_name = self.manipulator_name
        manipulator_map = self.manipulator_map
        t_idx = manipulator_map[manipulator_name]
        existing_manipulations = model_config['feature_settings']['manipulations']
        if t_idx == 0:
            updated_manipulations = expanded_transformations + existing_manipulations[1:]
            end_idx = len(expanded_transformations)
            prior_manipulator_offset = 0
        else:
            prior_manipulator_entry = existing_manipulations[t_idx - 1]
            prior_manipulator_name = list(prior_manipulator_entry.keys())[0]
            prior_manipulator = prior_manipulator_entry[prior_manipulator_name]['initialized_manipulator']
            prior_manipulator_offset = prior_manipulator.manipulator_map[prior_manipulator_name] + 1
            if t_idx < len(existing_manipulations) - 1:
                updated_manipulations = existing_manipulations[0:t_idx] + expanded_transformations + existing_manipulations[t_idx + 1:]
                end_idx = len(existing_manipulations[0:t_idx]) + len(expanded_transformations)
            else:
                updated_manipulations = existing_manipulations[:-1] + expanded_transformations
                end_idx = len(existing_manipulations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['manipulations'] = updated_manipulations
        updated_transformations = updated_manipulations[prior_manipulator_offset:end_idx]
        return updated_transformations

class interaction_terms(TransformChain):
    """
    Models:
      <model name>
        base_algorithm: <algorithm>
          feature_settings:
            manipulations:
              - interaction_terms:
                  interactions: (examples below)
                    - "('bill_sep','hist_sep')"
                    - "('bill_sep','prepay_sep')"
    """

    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        inter_terms_entry = self.fetch_manipulator_settings(model_config)
        raw_interaction_strs = inter_terms_entry['interactions']
        compact_interactions = [eval(ris) for ris in raw_interaction_strs]
        expanded_transformations = list()
        i = 0
        for tuple in compact_interactions:
            t0 = tuple[0]
            t1 = tuple[1]
            expanded_interactions = list(itertools.product([t0],[t1]))
            for inter in expanded_interactions:
                transformation_dict = dict()
                transformation_dict['int' + str(i) + '.' + 'ind_interaction_terms'] = {
                    'inclusion_patterns':[inter[0], inter[1]],
                    'kwargs' : {'include_bias': False, 'interaction_only': True}
                }
                expanded_transformations.append(transformation_dict)
                i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(interaction_terms, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class linear_combination(TransformChain):
    """
    linear_combination now supports simple mathematical operators (+,-,/,*), multiplicative or
    additive constants (as long as they're put after a variable), and sin/cos.
    Support must still be implemented for power-raising.

    keep_cols parameter (not required) determines whether columns specified in expression will be kept or
    dropped. Default is to keep. expression and equals parameters are required.

    sample yaml usage:
    ...
    manipulations:
        - linear_combination:
            expression: 'TotalBsmtSF + 1stFlrSF + 2ndFlrSF'
            equals: 'NewTestCol'
            keep_cols: False
    """
    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        lincomb_entry = self.fetch_manipulator_settings(model_config)
        expression = lincomb_entry['expression']
        #self.pattern = "[\+\*\/-]\s?\d(?:\s|\Z)|[\+\*\/-]|cos|sin"
        self.pattern = "[\+\*\/-]\s[\d\.]+|[\+\*\/-]|cos|sin"
        rel_cols = list(filter(lambda x: len(x) > 0, [x.strip('()\ ') for x in re.split(self.pattern, expression)]))
        operator_regex = re.compile(self.pattern)
        operator_raw_strs = operator_regex.findall(expression)
        expanded_transformations = list()
        i = 0
        for raw_str in operator_raw_strs:
            operator_type = self.determine_operator(raw_str)
            if operator_type == 'primal_op':
                inclusion_patterns = rel_cols[0:2]
                rel_cols = rel_cols[2:]
            else:
                inclusion_patterns = [rel_cols[0]]
                rel_cols = rel_cols[1:]
            if i + 1 == len(operator_raw_strs):
                col_name = lincomb_entry['equals']
                keep_cols = lincomb_entry['keep_cols']
            else:
                col_name = 'inter_quant_' + str(i)
                keep_cols = False
            transformation_dict = dict()
            transformation_dict['int' + str(i) + '.' + 'ind_' + operator_type] = {
                'inclusion_patterns': inclusion_patterns,
                'operator': raw_str.strip(),
                'equals': col_name,
                'keep_cols': keep_cols,
                'kwargs': {}
            }
            expanded_transformations.append(transformation_dict)
            rel_cols = [col_name] + rel_cols
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(linear_combination, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

    def determine_operator(self, raw_str):
        try:
            eval( str(1) + raw_str)
            return 'constant_op'
        except SyntaxError:
            pass
        if raw_str.strip() in ['+','-','/','*']:
            return 'primal_op'
        elif raw_str in ['cos','sin']:
            return 'trig_op'
        else:
            raise Exception

class box_cox_transform(TransformChain):

    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        transform_chain_settings = self.fetch_manipulator_settings(model_config)
        init_inclusion_patterns = transform_chain_settings['inclusion_patterns']
        skew_threshold = transform_chain_settings['skew_threshold']
        manipulations = model_config['feature_settings']['manipulations']
        num_box_cox_entries = len(filter(lambda x: 'ind_box_cox_transform' in x.keys()[0], manipulations))
        #todo: how do i fix this indexing thing again?
        bc_i = num_box_cox_entries + 1
        expanded_transformations = list()
        if init_inclusion_patterns in ['All Skewed', 'All Numeric']:
            num_tag_numeric_entries = len(filter(lambda x: 'tag_numeric' in x.keys()[0], manipulations))
            tn_i = num_tag_numeric_entries + 1
            tag_numeric_entry = { str(tn_i) + '.tag_numeric' : { 'inclusion_patterns': 'All'}}
            expanded_transformations = expanded_transformations + [tag_numeric_entry]
            if init_inclusion_patterns == 'All Skewed':
                num_tag_skewed_entries = len(filter(lambda x: 'tag_skewed' in x.keys()[0], manipulations))
                sk_i = num_tag_skewed_entries + 1
                tag_skewed_entry = { str(sk_i) + '.tag_skewed': {'inclusion_patterns' : 'All Numeric',
                                                                 'skew_threshold': skew_threshold}}
                expanded_transformations = expanded_transformations + [tag_skewed_entry]
            else:
                pass
            box_cox_entry = { str(bc_i) + '.ind_box_cox_transform': {'inclusion_patterns': init_inclusion_patterns,
                                                                     'kwargs': dict()}}
            expanded_transformations = expanded_transformations + [box_cox_entry]
        else:
            assert type(init_inclusion_patterns) == list
            for pattern in init_inclusion_patterns:
                box_cox_entry = { str(bc_i) + '.ind_box_cox_transform': { 'inclusion_patterns': [pattern],
                                                                          'kwargs': dict()}}
                expanded_transformations = expanded_transformations + [box_cox_entry]
                bc_i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(box_cox_transform, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class stack(TransformChain):
    """Example in models.yaml file:
        Models:
          Model Name:
            base_algorithm: sklearn.linear_model.LinearRegression #This is combo method outlined in ESL, Chapter 8.8
            feature_engineering:
                - stack:
                    algorithms:
                    - <fully qualified algo> (i.e. sklearn.ensemble.RandomForestRegressor)
                        keyword_arg_settings:
                            random_state: 1234  #(this is illustrative)
                    - <another fully qualified algo>:
                        keyword_arg_settings:
                            ...
        # At bottom of feature engineering section, be sure to include only the derived algo columns
                - include_features:
                    inclusion_patterns: [<fully qualified algo 1>(str) + '.hat',
                                        <fully qualified algo 2>(str) + '.hat']
    """
    def __init__(self, transform_chain_id, starting_transformations, model_config, project_settings):
        Manipulator.__init__(model_config, project_settings, starting_transformations)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in starting_transformations]
        t_idx = transformer_names.index('stack') #TODO: Figure out whether next 3 lines uniquely identify stacking entry?
        i = 0
        stack_entry = filter(lambda x: x.keys()[0] == 'stack', starting_transformations)[0]['stack']
        algorithms = stack_entry['algorithms']
        derived_cols = list()
        for algorithm_dict in algorithms:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_stack'
            algorithm_name = algorithm_dict.keys()[0]
            assert len(algorithm_dict) == 1
            if len(derived_cols) == 0:
                inclusion_patterns = ['All']
            else:
                inclusion_patterns = {'All But': derived_cols}
            transformation_dict[expanded_transformer_name] = {
                'algorithm': algorithm_dict.keys()[0],
                'inclusion_patterns': inclusion_patterns,
                'keyword_arg_settings' : algorithm_dict[algorithm_name]['keyword_arg_settings']
            }
            derived_cols = derived_cols + [algorithm_name + '.hat']
            expanded_transformations.append(transformation_dict)
            i += 1
        if t_idx == 0:
            starting_transformations = expanded_transformations + starting_transformations[1:]
            exp_idx = len(expanded_transformations)
        elif t_idx < len(starting_transformations) - 1:
            starting_transformations = starting_transformations[0:t_idx] + expanded_transformations + starting_transformations[t_idx + 1:]
            exp_idx = len(starting_transformations[0:t_idx]) + len(expanded_transformations)
        else:
            starting_transformations = starting_transformations[:-1] + expanded_transformations
            exp_idx = len(starting_transformations[:-1]) + len(expanded_transformations)
        model_config['feature_settings']['feature_engineering'] = starting_transformations
        super(stack, self).__init__(transformer_id, starting_transformations[:exp_idx], model_config)

class kaggle_stack(TransformChain):
    """example in yaml.file
  Model Name:
    base_algorithm: algorithms.common.MetaModeler
    feature_settings:
      manipulations:
        - kaggle_stack:
            inclusion_patterns:
              - 'All'
            validation_peeking: True/False
            algorithms:
              - sklearn.ensemble.RandomForestRegressor:
                  keyword_arg_settings:
                    random_state: 1234
              - sklearn.ensemble.GradientBoostingRegressor:
                  keyword_arg_settings:
                    random_state: 1234
    """
    def __init__(self, transform_chain_id, starting_transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        kaggle_stack_entry = self.fetch_manipulator_settings(model_config)
        expanded_transformations = list()
        transformer_names = [d.keys()[0] for d in starting_transformations]
        existing_include_feature_transformers = filter(lambda x: '.include_features' in x, transformer_names)
        if len(existing_include_feature_transformers) == 0:
            i = 0
        else:
            i = len(existing_include_feature_transformers)
        algorithms = kaggle_stack_entry['algorithms']
        derived_cols = [d.keys()[0] + '.hat' for d in algorithms]
        oos_predictor_ensemble_entry = dict()
        oos_predictor_ensemble_entry['oos_predictor_ensemble'] = {
            'algorithms': algorithms,
            'inclusion_patterns': ['All'],
            'validation_peeking': kaggle_stack_entry['validation_peeking']
        }
        expanded_transformations.append(oos_predictor_ensemble_entry)
        include_meta_features_transformer_settings = dict()
        include_meta_features_transformer_settings[str(i) + '.include_features'] = {
            'inclusion_patterns' : derived_cols
        }
        i += 1
        expanded_transformations.append(include_meta_features_transformer_settings)
        metamodel_transformer_settings = dict()
        metamodel_transformer_settings['metamodel'] = {
            'base_algorithm': 'sklearn.linear_model.LinearRegression',
            'keyword_arg_settings' : {},
            'inclusion_patterns': derived_cols
        }
        expanded_transformations.append(metamodel_transformer_settings)
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(kaggle_stack, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class as_numeric(TransformChain):
    """Example in models.yaml file:
    Models:
      <model name>:
       feature_settings:
         feature_engineering:
           - as_numeric
               val_maps:
                 - <pattern> : { str:val, str:val,...}
    """

    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        as_numeric_entry = self.fetch_manipulator_settings(model_config)
        val_maps = as_numeric_entry['val_maps']
        i = 0
        expanded_transformations = list()
        for pattern_val_map in val_maps:
            pattern = pattern_val_map.keys()[0]
            val_map = pattern_val_map[pattern]
            transformation_dict = dict()
            transformation_dict[str(i) + '.' + 'ind_as_numeric'] = {
                'inclusion_patterns': [pattern],
                'val_map': val_map
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(as_numeric, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class encode(TransformChain):
    """
    This transformation converts numeric variables to categorical ones
    TODO: Handle missing data, i.e. convert missing data into category. That would involve including this in pre-processing/data cleaning

    sample yaml usage:
    ...
    manipulations:
        - encode:
          inclusion_patterns:
            - 'YrSold'
    """
    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        encode_entry = self.fetch_manipulator_settings(model_config)
        inclusion_patterns = encode_entry['inclusion_patterns']
        i = 0
        expanded_transformations = list()
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            transformation_dict[str(i) + '.' + 'ind_encode'] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(encode, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class drop_outliers(TransformChain):
    """Example yaml
        ...
        feature_settings:
         manipulations:
          - drop_outliers:
                inclusion_patterns:
                    - <pattern>
                strategy: 'IQR'
        **NOTE: transformer doesn't currently do anything with strategy parameter. IQR is only strategy
                implemented so far
    """
    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        expanded_transformations = list()
        drop_outliers_entry = self.fetch_manipulator_settings(model_config)
        i = 0
        inclusion_patterns = drop_outliers_entry['inclusion_patterns']
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            expanded_transformer_name = '_' + str(i) + '.' + 'ind_drop_outliers'
            transformation_dict[expanded_transformer_name] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(drop_outliers, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class delete_vars(TransformChain, Cleaner):

    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        manipulations = model_config['feature_settings']['manipulations']
        existing_delete_vars_entries = filter(lambda x: 'ind_delete_var' in x.keys()[0], manipulations)
        delete_vars_entry = self.fetch_manipulator_settings(model_config)
        inclusion_patterns = delete_vars_entry['inclusion_patterns']
        chain_name_prefix = transform_chain_id.split('_')[0]
        i = 0 + len(existing_delete_vars_entries)
        expanded_transformations = list()
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            transformation_dict[chain_name_prefix + '_' + str(i) + '.' + 'ind_delete_var'] = {
                'inclusion_patterns': [pattern],
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(delete_vars, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class impute_vars(TransformChain, Cleaner):

    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        manipulations = model_config['feature_settings']['manipulations']
        existing_impute_vars_entries = filter(lambda x: 'ind_impute_var' in x.keys()[0], manipulations)
        impute_vars_entry = self.fetch_manipulator_settings(model_config)
        inclusion_patterns = impute_vars_entry['inclusion_patterns']
        replace_with = impute_vars_entry['replace_with']
        chain_name_prefix = transform_chain_id.split('_')[0]
        i = 0 + len(existing_impute_vars_entries)
        expanded_transformations = list()
        for pattern in inclusion_patterns:
            transformation_dict = dict()
            transformation_dict[chain_name_prefix +'_' + str(i) + '.' + 'ind_impute_var'] = {
                'inclusion_patterns': [pattern],
                'replace_with': replace_with
            }
            expanded_transformations.append(transformation_dict)
            i += 1
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(impute_vars, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)

class handle_missing_data(TransformChain):
    """***THIS TRANSFORMER MUST BE FIRST IN LIST OF MANIPULATIONS. DATA WILL BE DELETED IF NOT***
    example in yaml.file
    Model Name:
    base_algorithm: algorithms.common.MetaModeler
    feature_settings:
      manipulations:
        - handle_missing_data:
            - delete_vars:
                inclusion_patterns
            - delete_obs
                exclusion_patterns:
            - impute_vars
                inclusion_patterns:
                    - 'All Numeric'
                replace_with: 'mean'/'mode'
            - recode
                replace_with:
                    var_name: replacement_value
                    var_name: replacement_value

    """
    def __init__(self, transform_chain_id, transformations, model_config, project_settings):
        Manipulator.__init__(self, transform_chain_id, model_config, project_settings)
        handle_missing_data_settings = self.fetch_manipulator_settings(model_config)
        expanded_transformations = list()
        recode_entries = filter(lambda x: x.keys()[0] == 'recode', handle_missing_data_settings)
        if len(recode_entries) > 0:
            assert len(recode_entries) == 1
            recode_entry = recode_entries[0]
            inclusion_patterns = recode_entry['recode']['replace_with'].keys()
            existing_manipulations = model_config['feature_settings']['manipulations']
            num_existing_recoders = len(filter(lambda x: 'recode' in x.keys()[0], existing_manipulations))
            i = 0 + num_existing_recoders
            for pattern in inclusion_patterns:
                replace_with = recode_entry['recode']['replace_with'][pattern]
                base_col_name = pattern.split('_')[0]
                transformer_id = 'hmd_' + str(i) + '.recode'
                pattern_entry = { transformer_id : { 'inclusion_patterns': [base_col_name],
                                                     'val_map': { 'NA' : replace_with }
                                                    }
                                }
                expanded_transformations = expanded_transformations + [pattern_entry]
                i += 1
        impute_vars_entries = filter(lambda x: x.keys()[0] == 'impute_vars', handle_missing_data_settings)
        if len(impute_vars_entries) > 0:
            for i in range(len(impute_vars_entries)):
                impute_vars_sub_entry = impute_vars_entries[i]
                inclusion_patterns = impute_vars_sub_entry['impute_vars']['inclusion_patterns']
                replace_with = impute_vars_sub_entry['impute_vars']['replace_with']
                impute_vars_entry = { 'hmd_' + str(i) + '.impute_vars': { 'inclusion_patterns' : inclusion_patterns,
                                                                 'replace_with': replace_with}
                                      }
                expanded_transformations = expanded_transformations + [impute_vars_entry]
                i += 1
        #TODO: Should I alert user when an imputer does nothing? i.e. there's no missing values to impute?
        delete_vars_entries = filter(lambda x: x.keys()[0] == 'delete_vars', handle_missing_data_settings)
        if len(delete_vars_entries) > 0:
            assert len(delete_vars_entries) == 1
            delete_vars_sub_entry = delete_vars_entries[0]
            inclusion_patterns = delete_vars_sub_entry['delete_vars']['inclusion_patterns']
            delete_vars_entry = { 'hmd_0.delete_vars' : { 'inclusion_patterns': inclusion_patterns}}
            expanded_transformations = expanded_transformations + [delete_vars_entry]
        delete_obs_entries =  filter(lambda x: x.keys()[0] == 'delete_obs', handle_missing_data_settings)
        if len(delete_obs_entries) > 0:
            assert len(delete_obs_entries) == 1
            delete_obs_sub_entry = delete_obs_entries[0]
            delete_obs_entry = { 'hmd_0.delete_obs': {'inclusion_patterns': 'All'}}
            expanded_transformations = expanded_transformations + [delete_obs_entry]
        updated_transformations = self.update_manipulations_and_transformations(expanded_transformations)
        super(handle_missing_data, self).__init__(transform_chain_id, updated_transformations, model_config, project_settings)