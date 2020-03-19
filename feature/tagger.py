from scipy.stats import skew
from utils import flip_dict

from feature.engineering import Transformer
from .base_transformers import Identity

class Tagger(Transformer):

    def __init__(self, tagger_id, model_config, project_settings):
        super(Tagger, self).__init__(tagger_id, model_config, project_settings)

class VerticalTagger(Tagger):

    def __init__(self, tagger_id, model_config, project_settings):
        super(VerticalTagger, self).__init__(tagger_id, model_config, project_settings)

    def configure_features(self):
        prior_features = self.load_prior_features()
        touch_indices, untouched_indices = self.return_touch_untouched_indices(prior_features)
        self.touch_indices = touch_indices
        self.untouched_indices = untouched_indices
        self.features = prior_features

class tag_numeric(VerticalTagger):

    def __init__(self, tagger_id, model_config, project_settings):
        super(tag_numeric, self).__init__(tagger_id, model_config, project_settings)
        self.set_base_transformer(Identity(**self.kwargs))

    def fit(self, X_mat, y, **kwargs):
        prior_features = self.load_prior_features()
        non_inter_numeric_features = {k:v for k,v in prior_features.items() if '_' not in v}.values()
        inter_features = {k:v for k,v in prior_features.items() if 'x%x' in v}
        #TODO: Line below will only handle single order interactions, not interactions of interactions
        #TODO: Test that this approach actually works with interaction features
        numeric_inter_features = filter(lambda x: len(x.split('_')) < 4, inter_features.values())
        model_config = self.model_config
        model_config['numeric_features'] = non_inter_numeric_features + numeric_inter_features

class tag_skewed(VerticalTagger):

    def __init__(self, tagger_id, model_config, project_settings):
        super(tag_skewed, self).__init__(tagger_id, model_config, project_settings)
        tagger_settings = self.fetch_manipulator_settings(model_config)
        skew_threshold = tagger_settings['skew_threshold']
        self.skew_threshold = skew_threshold
        self.set_base_transformer(Identity(**self.kwargs))

    def fit(self, X_mat, y, **kwargs):
        skew_threshold = self.skew_threshold
        skew_scores = X_mat.apply(lambda x: abs(skew(x,**kwargs)))
        skewed_indices = list((skew_scores > skew_threshold).index)
        prior_features = self.load_prior_features()
        skewed_features = [prior_features[i] for i in skewed_indices]
        model_config = self.model_config
        model_config['skewed_features'] = skewed_features




