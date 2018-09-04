import pandas as pd
import numpy as np
import pickle
from src.DataCleaning2 import CleanClass, CleanErs
from sklearn.ensemble import RandomForestRegressor
import src.ModelFunctions as mf

if __name__ == "__main__":
    df_ers_ccqb = pd.read_excel('src/data/ers_ccqb.xlsx')
    df_ers_scores1 = pd.read_excel('src/data/ers_rating_scores.xlsx')
    df_ers_scores2 = pd.read_excel('src/data/ers_rating_scores.xlsx', sheet_name=1, header=1)
    df_class = pd.read_excel('src/data/class_ccqb.xlsx')
    df_class_scores = pd.read_excel('src/data/class_ratings_scores.xlsx', header=(0,1))

    print("Loaded exel files")

    ers_transformer = CleanErs()
    class_transformer = CleanClass()

    ers_ccqb_cleaned, ers_ratings_cleaned = ers_transformer.fit_clean(df_ers_ccqb, df_ers_scores1, df_ers_scores2)
    print("Cleaned ERS")
    class_ccqb_cleaned, class_ratings_cleaned = class_transformer.fit_clean(df_class, df_class_scores)
    print("Cleaned CLASS")

    with open("static/ers/ers_transform.pkl", "wb") as f:
        pickle.dump(ers_transformer, f)
    with open("static/class/class_transform.pkl", "wb") as f:
        pickle.dump(class_transformer, f)
    print("Transformers pickled")

    ers_set = mf.create_data_sets(ers_ccqb_cleaned, ers_ratings_cleaned)
    print("Created ERS data sets")
    class_set = mf.create_data_sets(class_ccqb_cleaned, class_ratings_cleaned)
    print("Created CLASS data sets")

    ers_models = mf.create_fit_models(ers_set)
    print("Fit all ERS models")
    class_models = mf.create_fit_models(class_set)
    print("Fit all CLASS models")

    with open("static/ers/ers_models.pkl", "wb") as f:
        pickle.dump(ers_models, f)
    print("Pickled ERS models")
    with open("static/class/class_models.pkl", "wb")as f:
        pickle.dump(class_models, f)
    print("Pickles CLASS models")
    print("attempting to load pickles")
    with open("static/ers/ers_models.pkl", "rb") as f:
        models_test_ers = pickle.load(f)
    print("Loaded ers pickle")
    print(models_test_ers)
    with open("static/ers/ers_models.pkl", "rb") as f:
        models_test_class = pickle.load(f)
    print("Loaded class models")
    print(models_test_class)

    print("Completed")
