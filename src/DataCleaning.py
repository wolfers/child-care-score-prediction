import numpy as np
import pandas as pd

def drop_text_cols(df, drop_cols_list):
    '''
    Takes a DataFrame and a list of columns to drop
    returns DataFrame with all columns dropped and 
    also drops any columns containing text
    '''
    df = df.drop(drop_cols_list, axis=1).copy()
    for col in df.columns:
        if "Feedback for Provider" in col or "Notes" in col or "FB for Provider" in col:
            df = df.drop(col, axis=1).copy()
    return df

def remove_useless_cols(df):
    '''
    removes any columns that contain only 0s from a dataframe
    '''
    for col in df.columns:
        if list(df[col].unique()) == [0]:
            df = df.drop(col, axis=1).copy()
    return df

class CleanClassCCQB():
    def __init__(self):
        self.drop_cols = ['Identifier', 'Regard for Child/Student Persp Feedback',
                                'Touchpoint: Owner Name', 'Provider: Assigned Staff',
                                'Touchpoint: Created Date', 'Touchpoint: Created By',
                                'Room Observed', 'Documentation Time', 'Touchpoint: ID']
        self.dummy_cols = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type']
        self.columns = []

    def fit_transform_train(self, df):
        ''' 
        Take in CLASS CCQB data as a pandas DataFrame
        record averages for columns, total columns,
        and then return a transformed DataFrame ready to be
        put into a model.
        '''
        #drop unused columns
        df = drop_text_cols(df, self.drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self.dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #calculate and record average
        #create columsn for missing values
        #fill NaNs
        #record all used columns
        self.columns = df.columns
        return df
    
    def transform(self, df):
        '''
        take in a DataFrame and return
        a ready to use df for a model prediction
        must have already fir the calss on some data to transform
        '''
        #drop unused columns
        df = drop_text_cols(df, self.drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self.dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #create columns for missing values
        #fill NaNs with averages
        return df

