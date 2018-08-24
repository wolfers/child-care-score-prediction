import pandas as pd
import numpy as np
import pickle
from DataCleaning import CleanClass, CleanErs
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    df_ers_ccqb = pd.read_excel('data/ers_ccqb.xlsx')
    df_ers_scores1 = pd.read_excel('data/ers_rating_scores.xlsx')
    df_ers_scores2 = pd.read_excel('data/ers_rating_scores.xlsx', sheet_name=1, header=1)
    df_class = pd.read_excel('data/class_ccqb.xlsx')
    df_class_scores = pd.read_excel('data/class_ratings_scores.xlsx', header=(0,1))

    from DataCleaning import CleanErs, CleanClass
    ers_transformer = CleanErs()
    class_transformer = CleanClass()

    X_ers, y_ers = ers_transformer.fit_transform_train(df_ers_ccqb, df_ers_scores1, df_ers_scores2)
    X_class, y_class = class_transformer.fit_transform_train(df_class, df_class_scores)

    forest_model_ers = RandomForestRegressor(n_estimators=1500, max_features="log2")
    forest_model_class = RandomForestRegressor(n_estimators=1500, max_features="log2")

    forest_model_ers.fit(X_ers, y_ers)
    forest_model_class.fit(X_class, y_class)

    with open('data/ers_model.pkl', 'wb') as f:
        pickle.dump(forest_model_ers, f)
    with open('data/class_model.pkl', 'wb') as f:
        pickle.dump(forest_model_class, f)