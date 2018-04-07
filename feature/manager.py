import pandas as pd
import inspect
import importlib
import os

from django.utils.text import slugify

from utils import find_data_dir, find_project_dir


class FeatureManager:

    def __init__(self, model_config, project_settings):
        """Default order is to engineer before selection"""
        if model_config['feature_settings']['eng_post_selection'] == True:
            self.select_first = True
        else:
            self.select_first = False
        self.transformations = model_config['feature_settings']['feature_engineering']
        self.filters = model_config['feature_settings']['feature_selection']

        num_transformations = len(self.transformations)
        num_filters = len(self.filters)
        if num_transformations > 0 or num_filters > 0:
            project_dir = find_project_dir(project_settings)
            artifact_dir = project_dir + '/model_artifacts'
            if not os.path.isdir(artifact_dir):
                os.makedirs(artifact_dir)
            self.artifact_dir = artifact_dir

        self.model_config = model_config
        self.project_settings = project_settings

        self.working_features = None
        #project_dir = find_project_dir(project_settings)

        column_names_filepath = self.load_working_file_filepath('feature_names')
        self.inv_column_map = self.load_inv_column_map(column_names_filepath)

    def load_inv_column_map(self, filepath):
        df = pd.read_csv(filepath, sep="\s+", engine='python', names=['col_name'])
        return pd.Series(df.index, index=df.col_name).to_dict()

    def load_working_file_filepath(self,data_ref):
        project_settings = self.project_settings
        assert project_settings.has_key('working_files')
        working_files = project_settings['working_files']
        filepath = working_files[data_ref]
        if self._is_fully_qualified_path(filepath):
            wf_abs_filepath = filepath
        else:
            data_dir = find_data_dir(project_settings)
            wf_abs_filepath = data_dir + '/' + filepath
        return wf_abs_filepath

    def _is_fully_qualified_path(self,path):
        project_settings = self.project_settings
        repo_loc = project_settings['repo_loc']
        if repo_loc in path:
            return True
        else:
            return False

    def select_features(self,X_train,y_train):
        selection_module = importlib.import_module('feature.selection')
        filters = self.filters
        model_config = self.model_config
        self._pass_y_train_to_fm(y_train)
        X_filt = X_train
        orig_inv_col_map = self.inv_column_map
        orig_col_map = {v: k for k, v in orig_inv_col_map.iteritems()}
        init_num_feats = len(X_train.columns) #this might need to be changed when no longer pandas
        i = 1
        if len(filters) < 1:
            pass
        else:
            for d in filters:
                filter_name = d.keys()[0]
                print "\t[Train] Performing model selection (" + str(i) + '/' + str(len(filters)) + "): " + filter_name
                filter_class = getattr(selection_module,filter_name)
                additional_args = self._get_args(filter_class)
                filter = filter_class(model_config)
                kwargs = dict()
                for arg in additional_args:
                    kwargs[arg] = getattr(self,arg)
                X_filt = filter.apply(X_filt,**kwargs)
                i += 1
            working_features = self.working_features
            if working_features == None:
                working_features = orig_col_map
            updated_col_map = {idx: working_features[idx] for idx in X_filt.columns.tolist()}
            self._set_working_features(updated_col_map)
            filt_num_feats = len(self.working_features)
            print "\tAfter model selection, number of features now " + str(filt_num_feats) +", down from " + str(init_num_feats) +". " + \
                "See project model_artifacts folder for more info."
            self._output_features('selected-features')
            self._update_working_data_feature_names_ref('selected-features')
        return X_filt

    def _pass_y_train_to_fm(self,y_train):
        self.y_train = y_train

    def _update_working_data_feature_names_ref(self, feature_version):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_file_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        project_settings = self.project_settings
        project_settings['working_files']['feature_names'] = artifact_dir + '/' + model_file_name_prefix + \
                                                             '-' + (feature_version.replace('_','-')) + '.txt'
        assert 1 == 1

    def _set_working_features(self, col_map):
        self.working_features = col_map #TODO: Fix broken filter, which puts this in as a list of indices

    def _output_features(self,feature_version):
        model_config = self.model_config
        model_name = model_config['model_name']
        model_file_name_prefix = slugify(model_name)
        artifact_dir = self.artifact_dir
        pd.Series(self.working_features).to_csv(artifact_dir + '/' + model_file_name_prefix + '-' + slugify(feature_version) + '.txt')

    def _get_args(self,filter_class):
        args = getattr(inspect.getargspec(filter_class.apply),'args')
        additional_args = args[2:]
        return additional_args

    def engineer_features(self,X_mat,train=True):
        if train == True:
            log_prefix = "[Train] "
        else:
            log_prefix = "[Test] "
        engineering_module = importlib.import_module('feature.engineering')
        transformations = self.transformations
        model_config = self.model_config
        i = 1
        if len(transformations) < 1:
            X_transform = X_mat
        else:
            working_features = self.working_features
            if working_features == None:
                inv_col_map = self.inv_column_map
                working_features = {v: k for k, v in inv_col_map.iteritems()}
            for d in transformations:
                transform_name = d.keys()[0]
                print "\t" + log_prefix + "Performing feature engineering (" + str(i) + '/' + str(len(transformations)) + "): " + transform_name
                transform_class = getattr(engineering_module,transform_name)
                transform = transform_class(model_config)
                X_touch, X_untouched = transform.split(X_mat, working_features)
                X_touched = transform.fit(X_touch)
                X_transform, updated_col_map = transform.combine_and_reindex(X_touched, X_untouched, working_features)
                self._set_working_features(updated_col_map)
                if train == True:
                    self._output_features(transform_name)
                    self._update_working_data_feature_names_ref(transform_name)
                i += 1
        return X_transform

    def filter_selected_features(self,X_mat):
        if getattr(self,'working_features') != 'None':
            working_features = self.working_features # { ind : feat_name }
            print "\t[Test] Filtering selected features"
            X_filt = X_mat.iloc[:,working_features.keys()]
            return X_filt
        else:
            return X_mat