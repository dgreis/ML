from wrapper import Wrapper

class Bagged_Classifier(Wrapper):

    #TODO: Implement this. This is currently a placeholder
    def __init__(self,other_options,**kwargs):
        base_estimator_name = kwargs['estimator']
        base_estimator_class = None
        super(Bagged_Classifier,self).__init__(base_estimator=Basic_SVM_Classifier(**{'C':10}),
                                   **kwargs)