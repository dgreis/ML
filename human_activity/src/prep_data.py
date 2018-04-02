import yaml

from utils import find_data_dir

#TODO: import this logic into main ML script?

global_settings = yaml.load(open('../../global_settings.yaml'))
project_settings = yaml.load(open('./project_settings.yaml'))  #TODO: direct to project settings in global settings

