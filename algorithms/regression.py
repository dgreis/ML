import numpy as np
import matplotlib.pyplot as plt

from wrapper import Wrapper
from sklearn import ensemble
from sklearn import tree
from django.utils.text import slugify

##NOTE: I believe you only need to implement regression algorithms here that require something more.
##      i.e. for RandomForestRegressor, there is the feature importances which is output

class RandomForestRegressor(Wrapper):

    def __init__(self, model_config, project_settings):
        base_algo_class = ensemble.RandomForestRegressor
        super(RandomForestRegressor, self).__init__(base_algo_class, model_config, project_settings)

    def gen_output(self):
        assert hasattr(self,'feature_importances_')
        importances = self.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.estimators_],
                     axis=0)
        #num_features = self.base_algorithm.n_features
        num_features = 15

        indices = np.argsort(importances)[::-1][:num_features]
        features = self.features
        feature_names = [features[i] for i in indices]

        # Print the feature ranking
        #print("Feature ranking:")

        #for f in range(num_features):
        #    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        f = plt.figure()
        plt.title("Feature importances")
        plt.bar(range(num_features), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(num_features), feature_names)
        plt.xticks(rotation=75)
        plt.tick_params(labelsize=3)
        plt.xlim([-1, num_features])

        artifact_dir = self.artifact_dir
        model_name = self.model_name
        f.savefig(artifact_dir + '/' + slugify(model_name) + 'feature-importance-plot.pdf')

class DecisionTreeRegressor(Wrapper):  ##TODO: Can this be taken out? Because it doesn't implement anything beyond OTB sklearn functionality?

    def __init__(self, model_config, project_settings, mode='algorithm'):
        base_algo_class = tree.DecisionTreeRegressor
        super(DecisionTreeRegressor, self).__init__(base_algo_class, model_config, project_settings, mode)