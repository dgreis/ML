

class Wrapper(object):

    def __init__(self, base_algorithm_class, other_options, **kwargs):
        self.base_algorithm = base_algorithm_class(**kwargs)
        if other_options.has_key('gen_output'):
            arg_val = other_options['gen_output']
            self.gen_output_flag = arg_val
        else:
            self.gen_output_flag = False

    def fit(self,X,y):
        return self.base_algorithm.fit(X,y)

    def predict(self,X):
        return self.base_algorithm.predict(X)

    def gen_output(self):
        raise NotImplementedError