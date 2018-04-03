import yaml
import pandas as pd
import os

from utils import find_data_dir, load_column_map

#TODO: import this logic into main ML script?

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = yaml.load(open('./' + global_settings['current_project'] + '/src/project_settings.yaml'))

#TODO: clean up stupid duplicated columns in this dataset

def main():

    data_dir = find_data_dir(global_settings,project_settings)
    processed_dir = data_dir + '/processed'
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    raw_files = project_settings['raw_files']

    ##Remove duplicate columns or alter their names
    column_names_filepath = data_dir + '/' + raw_files['feature_names']
    feat_df = pd.read_csv(column_names_filepath,sep="\s+",engine='python',names=['file_index','feature_name'])
    feat_df.sort_values('feature_name',inplace=True)
    feat_df['col_name_count'] = feat_df.groupby(['feature_name']).cumcount()+1
    feat_df['new_feature_name'] = feat_df['feature_name'].where(feat_df['col_name_count'] == 1
                                 ,other=feat_df['feature_name'] + '..' + feat_df['col_name_count'].astype(str)
    )


    feature_name_filepath = data_dir + '/' + project_settings['final_files']['feature_names']
    feat_df.sort_values('file_index',inplace=True)
    feat_df[['new_feature_name']].to_csv(feature_name_filepath,index=False,header=False)


