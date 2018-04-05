from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class Multinomial_Classifier(LogisticRegression):

    def __init__(self,**kwargs):
        LogisticRegression.__init__(self,**kwargs)

class Random_Forest_Classifier(RandomForestClassifier):

    def __init__(self,**kwargs):
        RandomForestClassifier.__init__(self,**kwargs)


class MultiLayer_Perceptron_NN_Classifier(MLPClassifier):

    def __init__(self,**kwargs):
        MLPClassifier.__init__(self,**kwargs)

class Basic_SVM_Classifier(SVC):

    def __init__(self,**kwargs):
        SVC.__init__(self,**kwargs)

class Gaussian_Naive_Bayes_Classifier(GaussianNB):

    def __init__(self,**kwargs):
        GaussianNB.__init__(self,**kwargs)

class Gradient_Boosting_Classifier(GradientBoostingClassifier):

    def __init__(self,**kwargs):
        GradientBoostingClassifier.__init__(self,**kwargs)