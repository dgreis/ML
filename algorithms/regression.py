from sklearn import linear_model

class Ridge_Regression(linear_model.Ridge):

    def __init__(self,**kwargs):
        linear_model.Ridge.__init(self,**kwargs)