import yaml
import pandas as pd
import os
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils import find_data_dir, configure_project_settings

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():

    data_dir = find_data_dir(project_settings)
    raw_files = project_settings['preprocessed_input_files']

    X_filepath = data_dir + '/' + raw_files['X']
    df = pd.read_csv(X_filepath,header=None)

    X = df.iloc[:,range(6)]
    y = df.iloc[:,6]

    data_dict = OrderedDict([
        ('buying' ,['vhigh','high','med','low']),
        ('maint', ['vhigh','high','med','low']),
        ('doors' , ['2','3','4','5more']),
        ('persons', ['2','4','more']),
        ('lug_boot', ['small','med','big']),
        ('safety', ['low','med','high']),
        #('class', ['unacc', 'acc', 'good', 'vgood'])
    ])

    col_idx = 0
    meta_int_val_map = OrderedDict()
    for col in data_dict.keys():
        col_vals = data_dict[col]
        val_int_map = dict([(col_vals[i],i) for i in range(len(col_vals))])
        X.iloc[:, col_idx] =X.iloc[:,col_idx].apply(lambda x: val_int_map[x]).copy() #TODO: fix the annoying pandas warning once and for all
        meta_int_val_map[col] = {v: k for k,v in val_int_map.items()}
        col_idx += 1


    enc = OneHotEncoder(sparse=False)
    Xe = pd.DataFrame(enc.fit_transform(X))
    feature_names = list()
    for col in meta_int_val_map:
        int_val_map = meta_int_val_map[col]
        for i in range(len(int_val_map)):
            dim_val = int_val_map[i]
            feature_name = col + '_' + dim_val
            feature_names.append(feature_name)
    assert len(feature_names) == Xe.shape[1]

    clean_input_files = project_settings['clean_input_files']
    feature_names_rel_filepath = clean_input_files['feature_names']
    feature_names_abs_filetpah = data_dir + '/' + feature_names_rel_filepath
    pd.Series(feature_names).to_csv(feature_names_abs_filetpah,index=False,header=None,sep="\t")

    split_perc = project_settings['test_train_split']
    numpy_arrays = train_test_split(
        Xe, y, test_size=split_perc, random_state=42)

    i = 0
    for dataset_name in ["X_test", "X_train", "y_test", "y_train"]:
        array_i = numpy_arrays[i]
        df_i = pd.DataFrame(array_i)
        rel_file_path = clean_input_files[dataset_name]
        abs_file_path = data_dir + '/' + rel_file_path
        df_i.to_csv(abs_file_path,header=None,sep="\t",index=False)
        i += 1



