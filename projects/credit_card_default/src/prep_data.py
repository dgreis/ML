from __future__ import division
import yaml
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from utils import find_data_dir, configure_project_settings

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = configure_project_settings(global_settings)

def main():

    data_dir = find_data_dir(project_settings)
    processed_dir = data_dir + '/processed'
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

    data_dir = find_data_dir(project_settings)
    raw_files = project_settings['raw_input_files']

    X_filepath = data_dir + '/' + raw_files['X']

    df = pd.read_csv(X_filepath,sep=",",skiprows=1)

    X = df.iloc[:,range(24)]
    y = df.iloc[:,24]
    #y = pd.Series(np.repeat(1,len(y_str))).where(y_str == 'yes', other = 0)

    data_dict = OrderedDict(
        [
            ('id', 'numeric'),
            ('amt_given_credit', 'numeric'),
            ('gender',['1','2']),
            ('education',['0','1','2','3','4','5','6']),
            ('marital',['0','1','2','3']),
            ('age', 'numeric'),
            ('hist_apr','numeric'),
            ('hist_may','numeric'),
            ('hist_jun', 'numeric'),
            ('hist_jul','numeric'),
            ('hist_aug','numeric'),
            ('hist_sep','numeric'),
            ('bill_sep','numeric'),
            ('bill_aug','numeric'),
            ('bill_jul','numeric'),
            ('bill_jun','numeric'),
            ('bill_may','numeric'),
            ('bill_apr','numeric'),
            ('prepay_sep','numeric'),
            ('prepay_aug','numeric'),
            ('prepay_jul','numeric'),
            ('prepay_jun','numeric'),
            ('prepay_may','numeric'),
            ('prepay_apr','numeric')
            ]
    )

    X.columns = data_dict.keys()

    col_idx = 0
    meta_int_val_map = OrderedDict()
    for col in data_dict.keys():
        col_vals = data_dict[col]
        if col_vals != 'numeric':
            val_int_map = dict([(col_vals[i],i) for i in range(len(col_vals))])
            X.iloc[:, col_idx] =X.iloc[:,col_idx].astype(str).apply(lambda x: val_int_map[x]).copy()
            meta_int_val_map[col] = {v: k for k,v in val_int_map.items()}
        else:
            meta_int_val_map[col] = 'numeric'
        col_idx += 1

    Xe = pd.DataFrame()
    col_idx = 0
    enc = OneHotEncoder(sparse=False)
    feature_names = list()
    for col in meta_int_val_map:
        if meta_int_val_map[col] == 'numeric':
            Xe.loc[:,col] = X.loc[:,col]
            feature_names.append(col)
            col_idx += 1
        else:
            input_array = np.array(list(X.loc[:,col])).reshape(-1, 1)
            Xt = pd.DataFrame(enc.fit_transform(input_array))
            Xt_w = Xt.shape[1]
            Xt.columns = [i + col_idx for i in range(Xt_w)]
            Xe = pd.merge(Xe, Xt, left_index=True, right_index=True)
            int_val_map = meta_int_val_map[col]
            for i in range(len(int_val_map)):
                dim_val = int_val_map[i]
                feature_name = col + '_' + dim_val
                feature_names.append(feature_name)
                col_idx += 1
    assert col_idx == Xe.shape[1]
    Xe.columns = range(col_idx)

    clean_input_files = project_settings['clean_input_files']
    feature_names_rel_filepath = clean_input_files['feature_names']
    feature_names_abs_filepath = data_dir + '/' + feature_names_rel_filepath
    pd.Series(feature_names).to_csv(feature_names_abs_filepath,index=False,header=None,sep="\t")

    split_perc = project_settings['train_test_split']
    s1_dfs = [pd.DataFrame(x) for x in train_test_split(Xe, y, test_size=split_perc, random_state=42)]
    s1_names = ["X_test", "X_train_val", "y_test", "y_train_val"]
    dd1 = dict(zip(s1_names, s1_dfs))
    for name in dd1:
        rel_file_path = clean_input_files[name]
        abs_file_path = data_dir + '/' + rel_file_path
        dd1[name].to_csv(abs_file_path,header=None,sep="\t",index=False)
    X_train_val, y_train_val = dd1['X_train_val'], dd1['y_train_val']

    num_folds = project_settings['assessment']['cv_num_folds']
    perc = 1 / num_folds
    s2_dfs = [pd.DataFrame(x) for x in train_test_split(X_train_val,y_train_val,test_size=perc)]
    s2_names = ["X_train","X_val","y_train","y_val"]
    dd2 = dict(zip(s2_names,s2_dfs))
    for name in dd2:
        rel_file_path = clean_input_files[name]
        abs_file_path = data_dir + '/' + rel_file_path
        dd2[name].to_csv(abs_file_path,header=None,sep="\t",index=False)