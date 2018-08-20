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

def separate_df(df, sep_list):
    df = df.sort_values('Coded ID').copy()
    df2 = pd.DataFrame(df['Coded ID'])
    for col in df.columns[1:]:
        if col in sep_list or "Scored?" in col:
            df2[col] = df[col]
            df = df.drop(col, axis=1)
    return df, df2

def average_values(df):
    return df.groupby('Coded ID').mean()

def take_latest_date(df):
    rows = []
    for _, group in df.groupby('Coded ID'):
        rows.append(group.sort_values('Date').iloc[0])
    return pd.DataFrame(rows)

def get_record_types(df):
    #make a dummy if a groupby contians values
    df = pd.get_dummies(df[['Coded ID', 'Touchpoint: Record Type']],
                   dummy_na=True, columns=['Touchpoint: Record Type'])
    return df.groupby('Coded ID').max()

def fill_nans(df):
    col_avg_dict = {}
    for col in df.columns:
        avg = df[col].mean()
        col_avg_dict[col] = avg
    return df.fillna(col_avg_dict), col_avg_dict

def combine_df(df1, df2):
    record_type = get_record_types(df2)
    df2 = take_latest_date(df2)
    df2 = df2.drop('Touchpoint: Record Type', axis=1)
    df2.set_index('Coded ID', inplace=True)
    for col in df1:
        df2[col] = df1[col]
    for col in record_type:
        df2[col] = record_type[col]
    return df2

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

class CleanErs():
    def __init__(self):
        self._drop_cols = ['Identifier','Touchpoint: Owner Name',
                          'Provider: Assigned Staff', 'Touchpoint: Created Date',
                          'Touchpoint: Created By', 'Room Observed', 'Touchpoint: ID']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care']
        #list used when seperating things during the cleaning of the ccqb data
        self._sep_list = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type', 'Date']
        self._ccqb_col_avg_dict = {}

    def _clean_ers_ccqb(self, df_ers):
        ers_df_test = drop_text_cols(df_ers, self._drop_cols)
        ers_df_test = ers_df_test.drop([2939, 2940, 2941, 2942, 2943, 2944, 2945])
        ers_avg, ers_not_avg = separate_df(ers_df_test, self._sep_list)
        ers_avg = average_values(ers_avg)
        ers_no_nans, self._ccqb_col_avg_dict = fill_nans(ers_avg)
        ers_df_test = combine_df(ers_no_nans, ers_not_avg)
        ers_df_test = self._convert_scored_to_binary(ers_df_test)
        ers_df_test = pd.get_dummies(ers_df_test, dummy_na=True, columns=self._dummy_cols)
        ers_df_test = remove_useless_cols(ers_df_test)
        return ers_df_test
    

    def _convert_scored_to_binary(self, df):
        '''
        Takes in a DataFrame and converts all the columns in the ERS
        data that indicate scoring from Yes and No to 1 and 0
        returns transformed df
        '''
        for col in df.columns:
            if 'Scored?' in col:
                df[col] = df[col].eq('Yes').mul(1)
        return df

    def fit_transform_train(self, df_ccqb, df_scores):
        '''
        Take in ERS data as pandas DataFrames
        record averages for columns
        and then return a transformed DataFrame ready to be
        put into a model.
        '''
        clean_ccqb_db = self._clean_ers_ccqb(df_ccqb)
        return clean_ccqb_db

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