import graphviz

from utils import flip_dict
from wrapper import Wrapper
from sklearn import tree
from django.utils.text import slugify

class Decision_Tree_Classifier(Wrapper):

    def __init__(self, model_config, project_settings):
        base_algo_class = tree.DecisionTreeClassifier
        super(Decision_Tree_Classifier, self).__init__(base_algo_class, model_config, project_settings)

    def gen_output(self):
        clf = self.base_algorithm
        inv_col_map = self.inv_column_map
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