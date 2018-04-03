from sklearn.linear_model import LogisticRegression

class Multinomial_Classifier(LogisticRegression):

    def __init__(self,**kwargs):
        LogisticRegression.__init__(self,**kwargs)
