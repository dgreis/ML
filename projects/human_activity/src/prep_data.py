from __future__ import division

import yaml
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from utils import find_data_dir, configure_project_settings

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():

    data_dir = find_data_dir(project_settings)
    processed_dir = data_dir + '/processed'
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    raw_files = project_settings['raw_input_files']

    ##Remove duplicate columns or alter their names
    column_names_filepath = data_dir + '/' + raw_files['feature_names']
    feat_df = pd.read_csv(column_names_filepath,sep="\s+",engine='python',names=['file_index','feature_name'])
    feat_df.sort_values('feature_name',inplace=True)
    feat_df['col_name_count'] = feat_df.groupby(['feature_name']).cumcount()+1
    feat_df['new_feature_name'] = feat_df['feature_name'].where(feat_df['col_name_count'] == 1
                                 ,other=feat_df['feature_name'] + '..' + feat_df['col_name_count'].astype(str)
    )


    feature_name_filepath = data_dir + '/' + project_settings['clean_input_files']['feature_names']
    feat_df.sort_values('file_index',inplace=True)
    feat_df[['new_feature_name']].to_csv(feature_name_filepath,index=False,header=False,quoting=False,sep="\t")


    ##Create Validation Set
    num_folds = project_settings['assessment']['cv_num_folds']
    perc = 1 / num_folds
    clean_input_files = project_settings['clean_input_files']
    X_train_val_filepath = find_data_dir(project_settings) + '/' + clean_input_files['X_train_val']
    y_train_val_filepath = find_data_dir(project_settings) + '/' + clean_input_files['y_train_val']
    X_train_val = pd.read_csv(X_train_val_filepath,sep="\s+",engine='python',header=None)
    y_train_val = pd.read_csv(y_train_val_filepath,sep="\s+",engine='python',header=None)
    s2_dfs = [pd.DataFrame(x) for x in train_test_split(X_train_val,y_train_val,test_size=perc)]
    s2_names = ["X_train","X_val","y_train","y_val"]
    dd2 = dict(zip(s2_names,s2_dfs))
    for name in dd2:
        rel_file_path = clean_input_files[name]
        abs_file_path = data_dir + '/' + rel_file_path
        dd2[name].to_csv(abs_file_path,header=None,sep="\t",index=False)


