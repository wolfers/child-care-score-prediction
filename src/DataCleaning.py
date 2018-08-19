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

def convert_scored_to_binary(df):
    '''
    Takes in a DataFrame and converts all the columns in the ERS
    data that indicate scoring from Yes and No to 1 and 0
    returns transformed df
    '''
    for col in df.columns:
        if 'Scored?' in col:
            df[col] = df[col].eq('Yes').mul(1)
    return df

class CleanClassCCQB():
    def __init__(self):
        self._drop_cols = ['Identifier', 'Regard for Child/Student Persp Feedback',
                                'Touchpoint: Owner Name', 'Provider: Assigned Staff',
                                'Touchpoint: Created Date', 'Touchpoint: Created By',
                                'Room Observed', 'Documentation Time', 'Touchpoint: ID']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type']
        self._columns = []

    def fit_transform_train(self, df):
        ''' 
        Take in CLASS CCQB data as a pandas DataFrame
        record averages for columns, total columns,
        and then return a transformed DataFrame ready to be
        put into a model.
        '''
        #drop useless rows at the end
        df = df.drop([2584, 2585, 2586,2587, 2588, 2589, 2590])
        #take the most recent ratings
        idx = df.groupby(['Coded ID'], sort=False)['Date'].transform(max) == df['Date']
        df = df[idx]
        #average any ratings that are taken on the same day
        #drop unused columns
        df = drop_text_cols(df, self._drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self._dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #calculate and record average
        #create columsn for missing values
        #fill NaNs
        #record all used columns
        self._columns = df.columns
        return df
    
    def transform(self, df):
        '''
        take in a DataFrame and return
        a ready to use df for a model prediction
        must have already fir the calss on some data to transform
        '''
        #drop unused columns
        df = drop_text_cols(df, self._drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self._dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #create columns for missing values
        #fill NaNs with averages
        return df

class CleanErsCCQB():
    def __init__(self):
        self._drop_cols = ['Identifier','Touchpoint: Owner Name',
                          'Provider: Assigned Staff', 'Touchpoint: Created Date',
                          'Touchpoint: Created By', 'Room Observed', 'Touchpoint: ID']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type']
        self._columns = []
    
    def fit_transform_train(self, df):
        ''' 
        Take in ERS CCQB data as a pandas DataFrame
        record averages for columns, total columns,
        and then return a transformed DataFrame ready to be
        put into a model.
        '''
        #drop useless rows at the end
        df = df.drop([2939, 2940, 2941, 2942, 2943, 2944, 2945])
        #take the most recent ratings
        idx = df.groupby(['Coded ID'], sort=False)['Date'].transform(max) == df['Date']
        df = df[idx]
        #average any ratings that are taken on the same day
        #drop unused columns
        df = drop_text_cols(df, self._drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self._dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #calculate and record average
        #create columsn for missing values
        #fill NaNs
        #record all used columns
        self._columns = df.columns
        return df

    def transform(self, df):
        '''
        take in a DataFrame and return
        a ready to use df for a model prediction
        must have already fir the calss on some data to transform
        '''
        #drop unused columns
        df = drop_text_cols(df, self._drop_cols)
        #create dummies
        df = pd.get_dummies(df, dummy_na=True, columns=self._dummy_cols)
        #remove any columns that contain only 0s after creating the dummy columns
        df = remove_useless_cols(df)
        #create columns for missing values
        #fill NaNs with averages
        return df