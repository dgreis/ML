

class exclusion_patterns:

    def __init__(self,model_config):
        feature_selection_settings = model_config['feature_settings']['feature_selection']
        for item in feature_selection_settings:
            if item.keys()[0] == 'exclusion_patterns':
                self.exclusion_patterns = item['exclusion_patterns']
            else:
                pass

    def apply(self, X_mat, inv_column_map):
        exclude_columns = list()
        col_names = inv_column_map.keys()
        exclusion_patterns = self.exclusion_patterns
        for pattern in exclusion_patterns:
            pat_exclude_columns = filter(lambda x: pattern in x, col_names)
            exclude_columns = exclude_columns + pat_exclude_columns
        exclude_indices = [int(inv_column_map[col_name]) for col_name in exclude_columns]
        X_mat_filt = X_mat.drop(axis=1,labels=exclude_indices)
        print "\tdue to exclusion patterns, dropped " + str(len(exclude_indices)) + " features"
        return X_mat_filt