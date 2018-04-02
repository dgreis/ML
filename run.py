import yaml

from utils import *

#pseudo-code:
global_settings = yaml.load(open('./global_settings.yaml'))

project_settings = load_project_settings(global_settings)
model_configs = load_model_configs(global_settings)

models = configure_models(model_configs, project_settings)


for model_name in models:
    data = prepare_final_model_dataset(model_configs['Models'][model_name],project_settings)
    X_train,y_train  = data['X_train'], data['y_train']
    model = models[model_name]
    model.fit(X_train,y_train)
    #algo.predict(y_train)

#m.label(y)
#score(m,y)

