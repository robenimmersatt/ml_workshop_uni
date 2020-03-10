import pandas as pd


def drop_missing_values_vise_versa(df1, df2, column_name):
    drop_condition = df1[column_name].isin(df2[column_name]) == False
    df1 = df1.drop(df1[drop_condition].index)
    drop_condition = df2[column_name].isin(df1[column_name]) == False
    df2 = df2.drop(df2[drop_condition].index)
    return df1, df2


def read_and_clean_data(x_file_path, y_file_path):
    x = pd.read_csv(x_file_path, header=0, index_col=0, sep=",", decimal=".", dtype={0: int})
    x['date'] = pd.to_datetime(x.date, format='%Y-%m-%d')
    x = x.drop('quantity', axis=1)
    x = x[x.date < '2018-09-01']

    y = pd.read_csv(y_file_path, header=0, index_col=0, sep=",", dtype={0: int, 1: int})
    y = y.sort_values('recipient').reset_index(drop=True)

    x, y = drop_missing_values_vise_versa(x, y, 'recipient')

    return x, y


def get_data_RNN():
    pass
