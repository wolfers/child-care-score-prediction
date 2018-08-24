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
        if "Feedback for Provider" in col or "Notes" in col or "FB for Provider" in col or "Scored?" in col:
            df = df.drop(col, axis=1).copy()
    return df

def separate_df(df, sep_list):
    '''
    seperate the DataFrame into two seperate DataFrame using the dep_list
    The intention is to use one for finding the averages and filling the nans in those columns
    '''
    df = df.sort_values('Coded ID')
    df2 = pd.DataFrame(df['Coded ID'])
    for col in df.columns[1:]:
        if col in sep_list:
            df2[col] = df[col]
    return df.set_index('Coded ID'), df2.set_index('Coded ID')

def combine_df(df1, df2, train=True):
    '''
    combines DataFrames together for use in 
    cleaning the ccqb data
    '''
    if train == True:
        df2 = take_earliest_date(df2)
    for col in df1:
        df2[col] = df1[col]
    return df2

def get_dates_at_thresh_and_mean(df):
    rows = []
    for _, group in df.groupby('Coded ID'):
        earliest_date = group['Date'].min()
        rows.append(group[(group['Date'] < earliest_date + np.timedelta64(60, 'D'))].mean())
    return pd.DataFrame(rows, df.index.unique())

def take_earliest_date(df):
    '''
    takes the first date that data was entered for the provider
    '''
    rows = []
    for _, group in df.groupby('Coded ID'):
        rows.append(group.sort_values('Date').iloc[0])
    return pd.DataFrame(rows)

def get_record_types(df):
    '''
    make a dummy if a groupby contians values
    '''
    df = pd.get_dummies(df['Touchpoint: Record Type'],
                   dummy_na=True, columns=['Touchpoint: Record Type'])
    return df.groupby('Coded ID').max()

def fill_nans_train(df):
    '''
    Find the average of each column and then fill the NaNs with the average
    '''
    col_avg_dict = {}
    for col in df.columns:
        avg = df[col].mean()
        col_avg_dict[col] = avg
    df = create_nan_dummies(df)
    return df.fillna(col_avg_dict), col_avg_dict

def fill_nans_input(df, col_avg_dict):
    df = create_nan_dummies(df)
    return df.fillna(col_avg_dict)

def create_record_cols(df, record_types):
    for col in record_types:
        df[col] = record_types[col]
    df = df.drop('Touchpoint: Record Type', axis=1)
    return df

def create_nan_dummies(df):
    '''
    create dummy columns for NaN values
    '''
    for col in df.columns:
        df[col+'_nan'] = pd.isnull(df[col])
    return df.copy()

def get_dummy_dict(df, dummy_list):
    '''
    create a dict whos keys are columns that will eb converted to dummies
    values are the possible values of that column
    '''
    dummy_dict = {}
    for col in dummy_list:
        dummy_dict[col] = df[col].unique()
    return dummy_dict

def make_dummies(df, dummy_dict):
    for key, value in dummy_dict:
        convert_list = []
        for column in value:
            if df[key] == column:
                convert_list.append({'_'.join([key, column]): 1})
            else:
                convert_list.append({"_".join([key, column]): 0})
        temp_df = pd.DataFrame(convert_list)
        #try using concat
        for col in temp_df:
            df[col] = temp_df[col]
    return df

def clean_scores_dates(scores, ccqb):
    rows = []
    for name, group in scores.groupby('Coded Provider ID'):
        if name in ccqb.index:
            date = ccqb.loc[name]['Date']
            dates_df = group[(group['Date_means'] > date)]
            if len(dates_df) != 0:
                date_row = dates_df.sort_values('Date').iloc[0]
                rows.append(date_row)
    return pd.DataFrame(rows)


def combine_finished_dfs( df_ccqb, df_scores):
    df = pd.concat([df_ccqb, df_scores], 1, 'inner')
    df['time_delta'] = df[['Date', 'Date_means']].diff(axis=1)['Date_means'] / np.timedelta64(1, 'D')
    return df.drop(['Date', 'Date_means'], axis=1)

class CleanClass():
    def __init__(self):
        self._drop_cols = ['Identifier', 'Regard for Child/Student Persp Feedback',
                                'Touchpoint: Owner Name', 'Provider: Assigned Staff',
                                'Touchpoint: Created Date', 'Touchpoint: Created By',
                                'Room Observed', 'Documentation Time', 'Touchpoint: ID',
                                'Emotional/Classroom Org Average', 'Instructional/Engaged Language Average',
                                'Travel Time']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care']
        self._sep_list = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type', 'Date']
        self._scores_drop = ['Date', 'Assessment Id']
        self._ccqb_col_avg_dict = {}
        self._dummy_dict = {}

    def _clean_class_ccqb(self, df):
        self._dummy_dict = get_dummy_dict(df, self._dummy_cols)
        df = df.drop([2584, 2585, 2586,2587, 2588, 2589, 2590])
        df = drop_text_cols(df, self._drop_cols)
        df_to_avg, df_saved_cols = separate_df(df, self._sep_list)
        df_transformed = get_dates_at_thresh_and_mean(df_to_avg)
        df_no_nans, self._ccqb_col_avg_dict = fill_nans_train(df_transformed)
        record_types = get_record_types(df_saved_cols)
        df = combine_df(df_no_nans, df_saved_cols)
        df = create_record_cols(df, record_types)
        df = pd.get_dummies(df, columns=self._dummy_cols)
        return df

    def _clean_class_scores(self, df_scores, df_ccqb):
        df_scores = df_scores.copy()
        df_scores.columns = ['_'.join(col).strip() for col in df_scores.columns.values]
        df_scores = df_scores.rename({'Unnamed: 0_level_0_Assessment Id': 'Assessment Id', 
                                'Unnamed: 1_level_0_Assessment Phase Desc': 'Assessment Phase Desc',
                                'Unnamed: 2_level_0_Assessment': 'Assessment',
                                'Unnamed: 3_level_0_Date': 'Date'}, axis='columns')
        df_scores['Coded Provider ID'] = df_scores.index
        df_scores['Date_means'] = pd.to_datetime(df_scores['Date'])
        df_scores = clean_scores_dates(df_scores, df_ccqb)
        df_means = df_scores[['Date_means', 'Coded Provider ID']]
        df_means = df_means.set_index('Coded Provider ID')
        df_scores = df_scores.set_index('Coded Provider ID')
        df_scores = df_scores.drop(self._scores_drop, axis=1)
        df_means['mean'] = df_scores.mean(axis=1)
        return df_means

    def fit_transform_train(self, df_ccqb, df_scores):
        '''
        Take in CLASS CCQB data as a pandas DataFrame
        record averages for columns, total columns,
        and then return a transformed DataFrame ready to be
        put into a model.
        '''
        clean_ccqb_df = self._clean_class_ccqb(df_ccqb)
        clean_scores_df = self._clean_class_scores(df_scores, clean_ccqb_df)
        df = combine_finished_dfs(clean_ccqb_df, clean_scores_df)
        df = df.drop(np.nan, axis=1)
        return df.drop('mean', axis=1), df['mean'].values
    
    def transform(self, df):
        '''
        This function would be written to transform whatever form the data comes in as
        It preps the data to be run through a fit model for prediction
        '''
        pass

class CleanErs():
    def __init__(self):
        self._scores_drop = ['Assessment', 'Site Region', 'Assessment Phase Name']
        self._drop_cols = ['Identifier','Touchpoint: Owner Name', 'Provider: Assigned Staff', 
                          'Touchpoint: Created Date', 'Touchpoint: Created By', 'Room Observed', 'Touchpoint: ID',
                          'ERS Scale Average', 'Subscale Average - Space & Furnishings',
                          'Subscale Average - Personal Care Routine', 'Subscale Average - Language-Reasoning',
                          'Subscale Average - Activities', 'Subscale Average - Interactions', 'Subscale Average - Program Structure']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care']
        #list used when seperating things during the cleaning of the ccqb data
        self._sep_list = ['Provider: Region', 'Provider: Type of Care', 'Touchpoint: Record Type', 'Date']
        self._ccqb_col_avg_dict = {}
        self._dummy_dict = {}
        self._scores_drop = ['Site Region', 'Assessment Phase Name', 'Date']

    def _clean_ers_ccqb(self, df_ers):
        self._dummy_dict = get_dummy_dict(df_ers, self._dummy_cols)
        df = drop_text_cols(df_ers, self._drop_cols)
        #data contains some messy columns at the end with no data, this is only for those.
        df = df.drop([2939, 2940, 2941, 2942, 2943, 2944, 2945])
        df_to_avg, df_saved_cols = separate_df(df, self._sep_list)
        df_transformed = get_dates_at_thresh_and_mean(df_to_avg)
        df_no_nans, self._ccqb_col_avg_dict = fill_nans_train(df_transformed)
        record_types = get_record_types(df_saved_cols)
        df = combine_df(df_no_nans, df_saved_cols)
        df = create_record_cols(df, record_types)
        df = pd.get_dummies(df, columns=self._dummy_cols)
        return df


    def _clean_ers_scores(self, df_scores1, df_scores2, df_ccqb):
        df_scores1 = df_scores1.set_index('Assessment Id')
        df_scores2 = df_scores2.set_index('Assessment Id')
        df_scores = pd.concat([df_scores1, df_scores2], 1, 'inner')
        df_scores['Date_means'] = pd.to_datetime(df_scores['Date'])
        df_scores = clean_scores_dates(df_scores, df_ccqb)
        df_means = df_scores[['Date_means', 'Coded Provider ID']]
        df_means = df_means.set_index('Coded Provider ID')
        df_scores = df_scores.set_index('Coded Provider ID')
        df_scores = df_scores.drop(self._scores_drop, axis=1)
        df_means['mean'] = df_scores.mean(axis=1)
        return df_means

    def fit_transform_train(self, df_ccqb, df_scores1, df_scores2):
        '''
        Take in ERS data as pandas DataFrames
        record averages for columns
        returns X and y
        '''
        clean_ccqb_df = self._clean_ers_ccqb(df_ccqb)
        clean_scores_df = self._clean_ers_scores(df_scores1, df_scores2, clean_ccqb_df)
        df = combine_finished_dfs(clean_ccqb_df, clean_scores_df)
        df = df.drop(np.nan, axis=1)
        return df.drop('mean', axis=1), df['mean'].values

    def transform(self, df):
        '''
        This function would be written to transform whatever form the data comes in as
        It preps the data to be run through a fit model for prediction
        '''
        pass