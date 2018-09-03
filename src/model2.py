import pandas as pd
import numpy as np
import pickle
from DataCleaning import CleanClass, CleanErs
from sklearn.ensemble import RandomForestRegressor
import ModelFunctions as mf

if __name__ == "__main__":
    df_ers_ccqb = pd.read_excel('data/ers_ccqb.xlsx')
    df_ers_scores1 = pd.read_excel('data/ers_rating_scores.xlsx')
    df_ers_scores2 = pd.read_excel('data/ers_rating_scores.xlsx', sheet_name=1, header=1)
    df_class = pd.read_excel('data/class_ccqb.xlsx')
    df_class_scores = pd.read_excel('data/class_ratings_scores.xlsx', header=(0,1))

    from DataCleaning2 import CleanErs, CleanClass
    ers_transformer = CleanErs()
    class_transformer = CleanClass()

    ers_ccqb_cleaned, ers_ratings_cleaned = ers_transformer.fit_clean(df_ers_ccqb, df_ers_scores1, df_ers_scores2)
    class_ccqb_cleaned, class_ratings_cleaned = class_transformer.fit_clean(df_class, df_class_scores)

    ers_set = mf.create_data_sets(ers_ccqb_cleaned, ers_ratings_cleaned)
    class_set = mf.create_data_sets(class_ccqb_cleaned. class_ratings_cleaned)

    ers_models = mf.create_fit_models(ers_set)
    class_models = mf.create_fit_models(class_set)

    mf.pickle_models(ers_models, "ers")
    mf.pickle_models(class_models, "class")
