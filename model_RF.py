from sklearn.ensemble import RandomForestClassifier

def RFClassifierModel():
    parameters = {'bootstrap': True,
                  'min_samples_leaf': 1, #3
                  'n_estimators': 5, #50
                  'min_samples_split': 2, #10
                  'max_features': 'auto', #sqrt
                  'max_depth': None, #6
                   'n_jobs' : -1,
                  'max_leaf_nodes': None}
    model = RandomForestClassifier(**parameters)
    return model