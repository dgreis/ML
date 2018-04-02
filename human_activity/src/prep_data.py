import pandas as pd
import yaml

from utils import find_data_dir

#TODO: import this logic into main ML script?

global_settings = yaml.load(open('../../global_settings.yaml'))
project_settings = yaml.load(open('./project_settings.yaml'))  #TODO: direct to project settings in global settings

data_dir = find_data_dir(global_settings,project_settings)

files = project_settings['files']

data = dict()
for f in files:
    filepath = data_dir + '/' + files[f]
    data[f] = pd.read_csv(filepath,sep="'\s+",engine='python')
