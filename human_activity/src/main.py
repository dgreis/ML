import yaml

from prep_data import data
from utils import build_algorithms

#pseudo-code:

#X_train,y_train  = data['X_train'], data['y_train']

algorithm_config = yaml.load(open('./algorithms.yaml')) #TODO: point this from global settings
project_settings = yaml.load(open('./project_settings.yaml'))
algorithms = build_algorithms(algorithm_config,project_settings)


for algo in ALGORITHMS:
    algo.fit(X_train,y_train)
    algo.predict(y_train)

m.label(y)
score(m,y)

