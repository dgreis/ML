# ML

My personal ML sandbox. Goal: to rapidly prototype POC sk-learn models from simple, modular yaml files. To get started, create a uniquely-named project sub-directory in the `projects` folder. 

#### Current Dev Branches:

- Add pytorch/keras model support
- AWS EC2 model training using docker

The program expects the following structure from any given project:

```
<project_name>
├── __init__.py
├── data
│   ├── processed
│   │   ├── X_test.txt
│   │   ├── X_train.txt
│   │   ├── X_train_val.txt
│   │   ├── X_val.txt
│   │   ├── features.txt
│   │   ├── y_test.txt
│   │   ├── y_train.txt
│   │   ├── y_train_val.txt
│   │   └── y_val.txt
└── src
    ├── __init__.py
    ├── models.yaml
    ├── prep_data.py
    └── project_settings.yaml
 ```
 _NOTE:_ Be sure to change the `repo_loc` in `global_settings.yaml` to the location of the git repository on your machine. 
 
 - `prep_data.py`: This is a file you write to do any custom pre-processing. It should generate all the files in the `processed` folder. The program will check for these files when it runs; if it doesn't find them, it will run `prep_data.py`.
 
 - `project_settings.yaml`: see `house_prices/src/project_settings.yaml` as a model. This file holds several project-specific parameters/
 
 - `models.yaml`: this is where most of the funcionality is driven. Again, see `house_prices/src/models.yaml` as an example. 
 
 All of the feature engineering/selection components that have been implemented are in the `./feature/selection.py`, `./feature/engineering.py`, and `feature/transformchain.py` files. In the class definitions are examples of usage within a `models.yaml` file.
 
 

