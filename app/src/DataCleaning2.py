import numpy as np
import pandas as pd

from src.ScoresDicts import ers_dict, class_dict


def drop_text_cols(df, drop_cols_list):
    '''
    Takes a DataFrame and a list of columns to drop
    returns DataFrame with all columns dropped and
    also drops any columns containing text
    '''
    df = df.drop(drop_cols_list, axis=1).copy()
    for col in df.columns:
        if "Feedback for Provider" in col or "Notes" in col or\
             "FB for Prov" in col or "Scored?" in col or\
             "FBfor Prov" in col:
            df = df.drop(col, axis=1).copy()
    return df


def separate_df(df, sep_list):
    '''
    seperate the DataFrame into two seperate DataFrame using the sep_list
    The intention is to use one for finding the averages
     and filling the nans in those columns
    while preserving the other columns
    '''
    df = df.sort_values('Coded ID')
    df2 = pd.DataFrame(df['Coded ID'])
    for col in df.columns[1:]:
        if col in sep_list:
            df2[col] = df[col]
    return df.set_index('Coded ID'), df2.set_index('Coded ID')


def combine_df(df1, df2, train=True):
    '''
    combines two dfs together
    if called for training data it also gets the earliest date
    '''
    if train:
        df2 = take_earliest_date(df2)
    for col in df1:
        df2[col] = df1[col]
    return df2


def get_dates_at_thresh(df):
    rows = []
    for _, group in df.groupby('Coded ID'):
        earliest_date = group['Date'].min()
        rows.append(group[(group['Date'] < earliest_date + np.timedelta64(100, 'D'))])
    return pd.concat(rows)


def mean_any_same_types(df):
    dfs = []
    for _, group in df.groupby(['Coded ID', 'Touchpoint: Record Type']):
        if len(group) > 1:
            region = group['Provider: Region'].values
            type_of_care = group['Provider: Type of Care'].values
            record_type = group['Touchpoint: Record Type'].values
            date = group['Date'].values
            temp_df = pd.DataFrame(group.mean().to_dict(), index=group['Coded ID'].values)
            temp_df['Provider: Region'] = region[0]
            temp_df['Provider: Type of Care'] = type_of_care[0]
            temp_df['Touchpoint: Record Type'] = record_type[0]
            temp_df['Date'] = date[0]
            dfs.append(temp_df)
        else:
            group = group.set_index('Coded ID')
            dfs.append(group)
    return pd.concat(dfs, sort=False)


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
    takes all the of the record types for a provider
    and creates a dummy columns that contain
    a 1 for any record type entered for the provider.
    '''
    df = pd.get_dummies(df['Touchpoint: Record Type'],
                        dummy_na=True, columns=['Touchpoint: Record Type'])
    return df.groupby('Coded ID').max()


def fill_nans_train(df):
    '''
    Find the mean of each column and then fill the NaNs with the mean
    '''
    col_avg_dict = {}
    for col in df.columns:
        if "Score" in col or 'Number' in col:
            avg = df[col].mean()
            col_avg_dict[col] = avg
    df = create_nan_dummies(df)
    return df.fillna(col_avg_dict), col_avg_dict


def create_record_cols(df, record_types):
    '''
    Gets the record columns and adds them to the DataFrame
    drops original Touchpoint: Record Type column
    '''
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
    create a dict whos keys are columns that will be converted to dummies
    values are the possible values of that column
    '''
    dummy_dict = {}
    for col in dummy_list:
        dummy_dict[col] = df[col].unique()
    return dummy_dict


def clean_scores_dates(scores, ccqb):
    '''
    creates and returns a new DataFrame that containes
    the most scoring that is closest to
    the baseline rating, excluding any that come
    before the baseline.
    '''
    rows = []
    for name, group in scores.groupby('Coded Provider ID'):
        if name in ccqb.index:
            date = ccqb.loc[name]['Date']
            if type(date) == pd.Series:
                date = date.min()
            dates_df = group[(group['Date'] > date)]
            if len(dates_df) != 0:
                date_row = dates_df.sort_values('Date').iloc[0]
                rows.append(date_row)
    return pd.DataFrame(rows)


def combine_finished_dfs(df_ccqb, df_scores):
    '''
    combines the DataFrames and creates a column
    for the amount of days between the baseline and the rating.
    '''
    for row in set(df_ccqb.index):
        try:
            df_ccqb.loc[row, 'Date_means'] = df_scores.loc[row, 'Date_means']
            df_ccqb.loc[row, 'mean'] = df_scores.loc[row, 'mean']
        except:
            df_ccqb = df_ccqb.drop(row)
    df_ccqb['time_delta'] = df_ccqb[['Date', 'Date_means']].diff(axis=1)['Date_means']\
        / np.timedelta64(1, 'D')
    return df_ccqb.drop(['Date', 'Date_means'], axis=1)


def create_score_col_names(df):
    '''
    Creates and returns a list of score columns
    '''
    score_col_names = []
    for col in df.columns.values:
        if 'Score' in col:
            score_col_names.append(col)
    return score_col_names


def create_transform_dummies(df, dummy_dict):
    '''
    create dummy columns for the tranform data,
    based on the fit data columns
    '''
    for key, value in dummy_dict.items():
        for col in value:
            if df[key].values == [col]:
                df[col] = 1
            else:
                df[col] = 0
        df = df.drop(key, axis=1)
    return df


def fill_nans_trans(df, avg_dict):
    '''
    create nan dummies for each column and
    fills in nans with the averages of the fit data
    '''
    for col, avg in avg_dict.items():
        if df[col].isnull().values[0]:
            df[col + '_nan'] = 0
            df[col] = avg
        else:
            df[col + '_nan'] = 1
    return df


def fill_zero_nan(df):
    '''
    fill the 0.0 floats in the data with NaNs
    The zeroes are probably just people that didn't
    score those things and marked them with zeroes
    '''
    return df.replace(0.0, np.nan)


def combine_and_rename(df, columns_dict):
    '''
    combineofficial ratings columns that are the 
    same and rename them to the columns they're 
    mapped to in the CCQB DataFrame
    '''
    df_copy = df.copy()
    for ccqb_col, score_cols in columns_dict.items():
        df_copy[ccqb_col] = df_copy[score_cols].mean(axis=1)
        df_copy = df_copy.drop(score_cols, axis=1)
    return df_copy


def remove_nan_cols(df):
    '''
    remove any columns that contain only NaNs
    '''
    for col in df.columns:
        if df[col].isnull().all():
            df = df.drop(col, axis=1)
    return df


class CleanClass():
    def __init__(self):
        self._drop_cols = ['Identifier',
                           'Regard for Child/Student Persp Feedback',
                           'Touchpoint: Owner Name',
                           'Provider: Assigned Staff',
                           'Touchpoint: Created Date',
                           'Touchpoint: Created By',
                           'Room Observed', 'Documentation Time',
                           'Touchpoint: ID',
                           'Emotional/Classroom Org Average',
                           'Instructional/Engaged Language Average',
                           'Travel Time', 'Observation Time']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care',
                            'Touchpoint: Record Type']
        self._sep_list = ['Provider: Region', 'Provider: Type of Care',
                          'Touchpoint: Record Type', 'Date']
        self._scores_drop = ['Date', 'Assessment Id', 'Assessment',
                             'Assessment Phase Desc']
        self._ccqb_col_avg_dict = {}
        self._dummy_dict = {}
        self.score_col_names = []

    def _clean_class_ccqb(self, df):
        '''
        clean the class ccqb data.
        drops useless rows at the end of excel sheet,
        creates a dict of dummy cols, drops text columns,
        finds the right dates and averages,
        returns cleaned ccqb data
        '''
        self._dummy_dict = get_dummy_dict(df, self._dummy_cols)
        df = df.drop([2584, 2585, 2586, 2587, 2588, 2589, 2590])
        df = drop_text_cols(df, self._drop_cols)
        self.score_col_names = create_score_col_names(df)
        df_dated = get_dates_at_thresh(df)
        df_averaged = mean_any_same_types(df_dated)
        df_no_nans, self._ccqb_col_avg_dict = fill_nans_train(df_averaged)
        df = pd.get_dummies(df_no_nans, columns=self._dummy_cols)
        return df

    def _clean_class_scores(self, df_scores, df_ccqb):
        '''
        cleans the class scores data
        flattens the muti-index DataFrame to one index,
        converts dates to datetime, cleans the data
        outputs a dataframe containing the dates and means
        '''
        df_scores = df_scores.copy()
        df_scores.columns = ['_'.join(col).strip() for
                             col in df_scores.columns.values]
        df_scores = df_scores.rename({'Unnamed: 0_level_0_Assessment Id':
                                      'Assessment Id',
                                      'Unnamed: 1_level_0_Assessment Phase Desc':
                                      'Assessment Phase Desc',
                                      'Unnamed: 2_level_0_Assessment': 'Assessment',
                                      'Unnamed: 3_level_0_Date': 'Date'},
                                     axis='columns')
        df_scores['Coded Provider ID'] = df_scores.index
        df_scores['Date'] = pd.to_datetime(df_scores['Date'])
        df_scores = clean_scores_dates(df_scores, df_ccqb)
        df_scores = df_scores.set_index('Coded Provider ID')
        df_scores = remove_nan_cols(df_scores)
        df_scores = fill_zero_nan(df_scores)
        df_scores = combine_and_rename(df_scores, class_dict)
        df_scores = df_scores.drop(self._scores_drop, axis=1)
        return df_scores

    def fit_clean(self, df_ccqb, df_scores):
        '''
        Take in data as pandas DataFrames
        returns cleaned CCQB and official ratings DataFrames
        '''
        clean_ccqb_df = self._clean_class_ccqb(df_ccqb)
        clean_scores_df = self._clean_class_scores(df_scores, clean_ccqb_df)
        return clean_ccqb_df.drop(['Date', 'Coded ID', 'Coded ID_nan', 
                                   'Date_nan', 'Provider: Region_nan', 
                                   'Provider: Type of Care_nan', 
                                   'Touchpoint: Record Type_nan'], axis=1), clean_scores_df

    def transform(self, df):
        '''
        Written to take in data from the webapp
        Takes in a dataframe, fills the nans,
        creates dummy columns
        '''
        df = fill_nans_trans(df, self._ccqb_col_avg_dict)
        df = create_transform_dummies(df, self._dummy_dict)
        return df.drop(np.nan, axis=1)


class CleanErs():
    def __init__(self):
        self._scores_drop = ['Assessment', 'Site Region',
                             'Assessment Phase Name']
        self._drop_cols = ['Identifier', 'Touchpoint: Owner Name',
                           'Provider: Assigned Staff',
                           'Touchpoint: Created Date',
                           'Touchpoint: Created By',
                           'Room Observed', 'Touchpoint: ID',
                           'ERS Scale Average',
                           'Subscale Average - Space & Furnishings',
                           'Subscale Average - Personal Care Routine',
                           'Subscale Average - Language-Reasoning',
                           'Subscale Average - Activities',
                           'Subscale Average - Interactions',
                           'Subscale Average - Program Structure']
        self._dummy_cols = ['Provider: Region', 'Provider: Type of Care',
                            'Touchpoint: Record Type']
        self._ccqb_col_avg_dict = {}
        self._dummy_dict = {}
        self._scores_drop = ['Site Region', 'Assessment Phase Name', 'Date',
                             'General supervision of children',
                             'Display for children', 'Assessment']
        self.score_col_names = []

    def _clean_ers_ccqb(self, df_ers):
        '''
        cleans ers ccqb data.
        creates dummy column dict, drops useless rows and columns,
        find right dates and averages
        returns cleaned data
        '''
        self._dummy_dict = get_dummy_dict(df_ers, self._dummy_cols)
        df = drop_text_cols(df_ers, self._drop_cols)
        df = df.drop([2939, 2940, 2941, 2942, 2943, 2944, 2945])
        self.score_col_names = create_score_col_names(df)
        df_dated = get_dates_at_thresh(df)
        df_averaged = mean_any_same_types(df_dated)
        df_no_nans, self._ccqb_col_avg_dict = fill_nans_train(df_averaged)
        df = pd.get_dummies(df_no_nans, columns=self._dummy_cols)
        return df

    def _clean_ers_scores(self, df_scores1, df_scores2, df_ccqb):
        '''
        takes in both scores DataFrames,
        concats them together, converts dates to datetime objects,
        returns cleaned and combined DataFrame
        '''
        df_scores1 = df_scores1.set_index('Assessment Id')
        df_scores2 = df_scores2.set_index('Assessment Id')
        df_scores = pd.concat([df_scores1, df_scores2], 1, 'inner')
        df_scores['Date'] = pd.to_datetime(df_scores['Date'])
        df_scores = df_scores.set_index('Coded Provider ID')
        df_scores = clean_scores_dates(df_scores, df_ccqb)
        df_scores = remove_nan_cols(df_scores)
        df_scores = fill_zero_nan(df_scores)
        df_scores = combine_and_rename(df_scores, ers_dict)
        df_scores = df_scores.drop(self._scores_drop, axis=1)
        return df_scores

    def fit_clean(self, df_ccqb, df_scores1, df_scores2):
        '''
        Take in ERS data as pandas DataFrames
        returns cleaned CCQB and official ratings DataFrames
        '''
        clean_ccqb_df = self._clean_ers_ccqb(df_ccqb)
        clean_scores_df = self._clean_ers_scores(df_scores1,
                                                 df_scores2,
                                                 clean_ccqb_df)
        return clean_ccqb_df.drop(['Date', 'Coded ID', 'Coded ID_nan', 
                                   'Date_nan', 'Provider: Region_nan', 
                                   'Provider: Type of Care_nan', 
                                   'Touchpoint: Record Type_nan'], axis=1), clean_scores_df

    def transform(self, df):
        '''
        Written to take in data from the webapp
        Takes in a dataframe, fills the nans,
        creates dummy columns
        '''
        df = fill_nans_trans(df, self._ccqb_col_avg_dict)
        df = create_transform_dummies(df, self._dummy_dict)
        return df.drop(np.nan, axis=1)
