import yaml
import os
import importlib

from utils import *

global_settings = yaml.load(open('./global_settings.yaml'))
project_settings = load_project_settings(global_settings)

if not all_final_files_exist(project_settings, global_settings):
    prep_data = importlib.import_module(project_settings['project_name'] + '.src.' + 'prep_data')
    prep_data.main()

model_configs = load_model_configs(global_settings)
models = configure_models(model_configs, project_settings)
num_models = len(models)

print "\nFit ML models for project: " + project_settings['project_name']
print "Number of models to fit: " + str(num_models)
i = 1
for model_name in models:
    print "\nFitting model (" + str(i) + "/" + str(num_models) + "): " + model_name +'. \nStep One: Finalize dataset'
    data = prepare_final_model_dataset(model_configs['Models'][model_name],project_settings)
    print "Data Finalized. \nStep Two: Fit Model"
    X_train,y_train  = data['X_train'], data['y_train']
    model = models[model_name]
    model.fit(X_train,y_train)
    print 'Model Fit. \nStep Three: Perform Model Evaluation'
    #algo.predict(y_train)
    i += 1

#m.label(y)
#score(m,y)

