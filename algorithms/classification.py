import graphviz

from utils import flip_dict, load_inv_column_map
from wrapper import Wrapper
from sklearn import tree
from django.utils.text import slugify

class DecisionTreeClassifier(Wrapper):

    def __init__(self, model_config, project_settings, mode='algorithm'):
        base_algo_class = tree.DecisionTreeClassifier
        super(DecisionTreeClassifier, self).__init__(base_algo_class, model_config, project_settings, mode)

    def gen_output(self):
        clf = self.base_algorithm
        prior_feature_names_filepath = self.prior_manipulator_feature_names_filepath
        inv_col_map = load_inv_column_map(prior_feature_names_filepath)
        col_map = flip_dict(inv_col_map)
        num_cols = len(col_map)
        column_names = [col_map[i] for i in range(num_cols)]
        dot_data = tree.export_graphviz(clf, out_file=None,feature_names=column_names)
        graph = graphviz.Source(dot_data,format='png')
        artifact_dir = self.artifact_dir
        model_name = self.model_name
        graph.render(filename=artifact_dir +'/' + slugify(model_name) + '-tree' )


class Bagged_Classifier(Wrapper):

    #TODO: Implement this. This is currently a placeholder
    def __init__(self,other_options,**kwargs):
        base_estimator_name = kwargs['estimator']
        base_estimator_class = None
        super(Bagged_Classifier,self).__init__(base_estimator=Basic_SVM_Classifier(**{'C':10}),
                                   **kwargs)