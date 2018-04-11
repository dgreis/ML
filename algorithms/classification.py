import importlib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class SKLearnWrapper(object):

    def __init__(self, sklearn_algo_name, other_options, **kwargs):
        this_mod = importlib.import_module('algorithms.classification')
        base_algorithm_class = getattr(this_mod, sklearn_algo_name)
        self.base_algorithm = base_algorithm_class(**kwargs)
        if other_options.has_key('gen_output'):
            arg_val = other_options['gen_output']
            self.gen_output_flag = arg_val
        else:
            self.gen_output_flag = False
        #super(Wrapper,self).__init__(**kwargs)

    def fit(self,X,y):
        return self.base_algorithm.fit(X,y)

    def predict(self,X):
        return self.base_algorithm.predict(X)

    def gen_output(self):
        raise NotImplementedError

class Logistic_Classifier(SKLearnWrapper):

    def __init__(self, other_options, **kwargs):
        sklearn_algo_name = 'LogisticRegression'
        super(Logistic_Classifier, self).__init__(sklearn_algo_name, other_options,**kwargs)
        assert 1 == 1


class Random_Forest_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(Random_Forest_Classifier,self).__init__(**kwargs)


class MultiLayer_Perceptron_NN_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(MultiLayer_Perceptron_NN_Classifier,self).__init__(**kwargs)

class Basic_SVM_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(Basic_SVM_Classifier,self).__init__(**kwargs)

class Gaussian_Naive_Bayes_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(Gaussian_Naive_Bayes_Classifier,self).__init__(**kwargs)

class Gradient_Boosting_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(Gradient_Boosting_Classifier,self).__init__(**kwargs)

class Bagged_SVM_Classifier(SKLearnWrapper):

    def __init__(self,**kwargs):
        super(Bagged_SVM_Classifier,self).__init__(base_estimator=Basic_SVM_Classifier(**{'C':10}),
                                   **kwargs)
