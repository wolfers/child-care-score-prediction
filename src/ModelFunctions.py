import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle


def create_data_sets(ccqb, scores):
    '''
    creates a dictionary of DataFrames. Each key is a column name
    in the scores DataFrame, each value is the ccqb DataFrame with
    the scores column for that key added as a target column
    '''
    data_sets = {}
    for col in scores.columns.values:
        df_copy = ccqb.copy()
        for row in set(df_copy.index):
            if row in scores.index:
                df_copy.loc[row, 'target'] = scores.loc[row, col]
            else:
                df_copy = df_copy.drop(row)
        df_copy = df_copy[(np.invert(df_copy['target'].isnull()))]
        data_sets[col] = df_copy
    return data_sets

def create_fit_models(data_sets):
    '''
    creates a dictionary where each key value pair
    is a name and a fit Random Forest Regressor for 
    each key, value pair in data_sets
    '''
    models_dict = {}
    for name, data in data_sets.items():
        forest_model = RandomForestRegressor(1000, max_features="log2")
        X = data.drop(['target'], axis=1)
        y = data['target'].values
        forest_model.fit(X, y)
        models_dict[name] = forest_model
    return models_dict

def pickle_models(models_dict, test):
    '''
    creates a pickled model for each model in
    models_dict. Also creates a pickled list
    containing the names of all the models
    '''
    if test == "ers":
        path = "child-care-score-prediction/app/static/ers/"
    else:
        path = "child-care-score-prediction/app/static/class/"
    for name, model in models_dict.items():
        with open(path + name + ".pkl", "wb") as f:
            pickle.dump(model, f)
    with open(path + "model_names.pkl", "wb") as f:
        pickle.dump(list(models_dict.keys()), f)


def load_pickled_models(test_type):
    '''
    load all the pickled models for the specified test
    and return them in a diction where the keys are the names
    and the values are the models
    '''
    pass

def predict_scores(dict_models, ccqb):
    '''
    take a ccqb and create a prediction for each model in the dictionary.
    return a dictionary of all the predicted models and their names
    '''
    dict_ratings = {}
    for name, model in dict_models.items():
        y = model.predict(ccqb)
        dict_ratings[name] = y
    return dict_ratings

def find_variance(dict_ratings, ccqb):
    '''
    take all of the ratings and find their varience from the original baseline.
    return a dict of the varience for each item.
    '''
    pass

def determine_top_changes(dict_variance, n=10):
    '''
    find the top n variances for the ratings and return them.
    '''
    pass

def process_ccqb(ccqb):
    '''
    get results for an inputted ccqb
    '''
    dict_ratings = predict_scores(ccqb)
    dict_variance = find_variance(dict_ratings, ccqb)
    dict_top = determine_top_changes(dict_variance, n=10)
    return dict_top