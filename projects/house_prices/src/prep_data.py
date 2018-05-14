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

    train_filepath = data_dir + '/' + raw_files['train']
    test_filepath = data_dir + '/' + raw_files['test']

    #for filepath in [train_filepath,test_filepath]: TODO: Figure this out
    df = pd.read_csv(train_filepath,sep=",",header=None,skiprows=1,keep_default_na=False)

    df = df.drop(64,axis=1)
    df.columns = range(df.shape[1])  #possible duplicate?

    #df = df.drop([0,38,46],axis=1) #Drop lincombo and id column
    #df.columns = range(df.shape[1])  #possible duplicate?

    X = df.iloc[:,range(df.shape[1]-1)]
    y = df.iloc[:,df.shape[1]-1]
    #y = pd.Series(np.repeat(1,len(y_str))).where(y_str == 'yes', other = 0)

    data_dict = OrderedDict(
        [
            ('id','numeric'),
            ('MSSubClass', ['20','30','40','45','50','60','70','75','80',
                            '85','90','120','160','180','190']), #Got rid of 150
            ('MSZoning', ['C (all)','FV','RH','RL','RM']), #Got rid of RP,I,C,A
            ('LotFrontage','numeric'),
            ('LotArea','numeric'),
            ('Street', ['Grvl', 'Pave']),
            ('Alley',['Grvl','Pave','NA']),
            ('LotShape', ['Reg','IR1','IR2','IR3']),
            ('LandContour',['Lvl','Bnk','HLS','Low']),
            ('Utilities',['AllPub','NoSeWa']), #Dropped NoSewr,ELO
            ('LotConfig', ['Inside','Corner','CulDSac','FR2','FR3']),
            ('LandSlope', ['Gtl', 'Mod', 'Sev']),
            ('Neighborhood',['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',
                             'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV',
                             'Mitchel','NAmes','NoRidge','NPkVill','NridgHt',
                             'NWAmes','OldTown','SWISU','Sawyer','SawyerW',
                             'Somerst','StoneBr','Timber','Veenker']),
            ('Condition1',['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA',
                           'RRNe','RRAe']),
            ('Condition2',['Artery','Feedr','Norm','RRNn','RRAn','PosA',
                           'PosN','RRAe']), #dropped RRNe
            ('BldgType',['1Fam','2fmCon','Duplex','TwnhsE','Twnhs']), #Does not match data description fi
            ('HouseStyle',['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf',
                         'SFoyer','SLvl']),
            ('OverallQual','numeric'),
            ('OverallCond', 'numeric'),
            ('YearBuilt','numeric'),
            ('YearRemodAdd','numeric'),
            ('RoofStyl',['Flat','Gable','Gambrel','Hip','Mansard','Shed']),
            ('RoofMatl',['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv',
                         'WdShake','WdShngl']),
            ('Exterior1st',['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd',
                            'HdBoard','ImStucc','MetalSd','Plywood',
                            'Stone','Stucco','VinylSd','Wd Sdng','WdShing']), #Dropped Other,PreCast
            ('Exterior2nd',['AsbShng','AsphShn','Brk Cmn','BrkFace','CBlock','CmentBd',
                            'HdBoard','ImStucc','MetalSd','Other','Plywood', #Dropped PreCast
                            'Stone','Stucco','VinylSd','Wd Sdng','Wd Shng']), #Does not match data desc file
            ('MasVnrType',['BrkCmn','BrkFace','None','Stone','NA']), #Does not match data desc file (NA) #Dropped CBlock
            ('MasVnrArea','numeric'),
            ('ExterQual',['Ex','Gd','TA','Fa']), #TODO: Make this numeric? #Dropped Po
            ('ExterCond',['Ex','Gd','TA','Fa','Po']),
            ('Foundation',['BrkTil','CBlock','PConc','Slab','Stone','Wood']),
            ('BsmtQual',['Ex','Gd','TA','Fa','NA']), #Dropped Po
            ('BsmtCond',['Gd','TA','Fa','Po','NA']), #Dropped Ex
            ('BsmtExposure',['Gd','Av','Mn','No','NA']),
            ('BasmtFinType1',['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']),
            ('BsmtFinSF1','numeric'),
            ('BsmtFinType2',['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']),
            ('BsmtFinSF2','numeric'),
            ('BsmtUnfSF','numeric'),
            ('TotalBsmtSF','numeric'),
            ('Heating',['Floor','GasA','GasW','Grav','OthW','Wall']),
            ('HeatingQC',['Ex','Gd','TA','Fa','Po']),
            ('CentralAir',['N','Y']),
            ('Electrical',['SBrkr','FuseA','FuseF','FuseP','Mix','NA']), #Does not match data desc file (NA)
            ('1stFlrSF','numeric'),
            ('2ndFlrSF','numeric'),
            ('LowQualFinSF','numeric'),
            ('GrLivArea','numeric'),
            ('BsmtFullBath','numeric'), #TODO: should these be cat?
            ('BsmtHalfBath','numeric'),
            ('FullBath','numeric'),
            ('HalfBath','numeric'),
            ('Bedroom','numeric'),
            ('Kitchen','numeric'),
            ('KitchenQual',['Ex','Gd','TA','Fa']), #Dropped Po
            ('TotRmsAbvGrd','numeric'),
            ('Functional',['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev']), #TODO: make numeric?  #dropped Sal
            ('Fireplaces','numeric'),
            ('FireplaceQu',['Ex','Gd','TA','Fa','Po','NA']),
            ('GarageType',['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd',
                           'NA']), #Does not match data desc file
            ('GarageYrBlt','numeric'),
            ('GarageFinish',['Fin','RFn','Unf','NA']),
            ('GarageCars','numeric'),
            ('GarageArea','numeric'),
            ('GarageCond',['Ex','Gd','TA','Fa','Po','NA']),
            ('PavedDrive',['Y','P','N']),
            ('WoodDeckSF','numeric'),
            ('OpenPorchSF','numeric'),
            ('EnclosedPorch','numeric'),
            ('3SsnPorch','numeric'),
            ('ScreenPorch','numeric'),
            ('PoolArea','numeric'),
            ('PoolQC',['Ex','Gd','Fa','NA']), #Dropped TA
            ('Fence',['GdPrv','MnPrv','GdWo','MnWw','NA']),
            ('MiscFeature',['Gar2','Othr','Shed','TenC','NA']), #dropped Elev
            ('MiscVal','numeric'),
            ('MoSold',['1','2','3','4','5','6','7','8','9','10','11','12']), #TODO: numeric?
            ('YrSold','numeric'),
            ('SaleType',['WD','CWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']), #dropped VWD
            ('SaleCondition',['Normal','Abnorml','AdjLand','Alloca','Family','Partial']),
            #('SalePrice','numeric')
             ]
    )

    X.columns = data_dict.keys()

    #Bad Formatting in source file
    #X.loc[:,'BldgType'] = X.loc[:,'BldgType'].copy().where(X.loc[:,'BldgType'] != '2fmCon',other='2FmCon')
    #X.loc[:,'BldgType'] = X.loc[:,'BldgType'].copy().where(X.loc[:,'BldgType'] != 'Duplex',other='Duplx')


    col_idx = 0
    meta_int_val_map = OrderedDict()
    for col in data_dict.keys():
        col_vals = data_dict[col]
        if col_vals != 'numeric':
            val_int_map = dict([(col_vals[i],i) for i in range(len(col_vals))])
            try:
                X.iloc[:, col_idx] =X.iloc[:,col_idx].astype(str).apply(lambda x: val_int_map[x]).copy()
            except KeyError:
                assert 1 == 0
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
            #try:
            #    assert 'NA' not in pd.isnull(X.loc[:,col].value_counts().index)
            #except AssertionError:
            #    assert 1 == 0
            X.loc[:,col] = X.loc[:,col].copy().apply(lambda x: x if x != 'NA' else np.nan)
            Xe.loc[:,col_idx] = X.loc[:,col].astype(float)
            Xe.loc[:,col_idx] = Xe.loc[:,col_idx].copy().fillna(Xe.loc[:,col_idx].mean()) #TODO: Make this a transfomer
            feature_names.append(col)
            col_idx += 1
        else:
            input_array = np.array(list(X.loc[:,col])).reshape(-1, 1)
            Xt = pd.DataFrame(enc.fit_transform(input_array))
            Xt_w = Xt.shape[1]
            Xt.columns = [i + col_idx for i in range(Xt_w)]
            if Xe.shape[1] == 0:
                Xe = Xt
            else:
                Xe = pd.merge(Xe, Xt, left_index=True, right_index=True)
            int_val_map = meta_int_val_map[col]
            try:
                assert len(int_val_map) == Xt_w
            except AssertionError:
                assert 1 == 0
            for i in range(len(int_val_map)):
                dim_val = int_val_map[i]
                feature_name = col + '_' + dim_val
                feature_names.append(feature_name)
                col_idx += 1
    assert col_idx == Xe.shape[1]
    Xe.columns = range(col_idx)

    feature_names = [x.replace(' ','_').replace('(','').replace(')','') for x in feature_names]

    clean_input_files = project_settings['clean_input_files']
    feature_names_rel_filepath = clean_input_files['feature_names']
    feature_names_abs_filepath = data_dir + '/' + feature_names_rel_filepath
    pd.Series(feature_names).to_csv(feature_names_abs_filepath,index=False,header=None,sep="\t")

    split_perc = project_settings['train_test_split']
    if split_perc != 1:
        s1_dfs = [pd.DataFrame(x) for x in train_test_split(Xe, y, test_size=split_perc, random_state=42)]
        s1_names = ["X_test", "X_train_val", "y_test", "y_train_val"]
    else:
        s1_dfs = [Xe,y]
        s1_names = ["X_train_val","y_train_val"]
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