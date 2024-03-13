import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import json

def fix_sys_dias(bp):
    '''
    the values in the systolic and diastolic columns are mixed up and erroneous i.e., systolic should be always greater than diastolic but due to manual entry
    some of the people entered it the other way. Moreover, some people have entered wrong data like 8.0 for diastolic etc.
    The code below will swap the diastolic and systolic values where required and remove the entries which falls below the specified range
    '''
    print('unfiltered shape', bp.shape)

    # Create a mask where diastolic is greater than systolic
    mask = bp['diastolic'] > bp['systolic']

    # Use the mask to swap the values
    bp.loc[mask, ['systolic', 'diastolic']] = bp.loc[mask, ['diastolic', 'systolic']].values
    print('number of sys-dias swaps', mask.sum())
    bp = bp[(bp['systolic'] >= 40) & (bp['systolic'] <= 340)]     # keeping only within range systolic values
    print('Shape of bp table after removing out of range systolic values', bp.shape)
    bp = bp[(bp['diastolic'] >= 10) & (bp['diastolic'] <= 200)]     # keeping only within range diastolic values
    print('Shape of bp table after removing out of range diastolic values', bp.shape)

    return bp

def master_merge(predictor_df, bp):
  '''
  Merges the predictor_df with the bp dataframe on healthCode and date
  '''
  bp['date'] = pd.to_datetime(bp['date'])
  predictor_df['date'] = predictor_df['date'].astype(bp['date'].dtypes)
  predictor_df = predictor_df.drop_duplicates(subset=['healthCode', 'date'])
  master_df = bp.merge(predictor_df, on=['healthCode', 'date'], how='left')
  return master_df


def get_non_zero(df):
    '''
    Gets a count of non-zero and non-NaN values in the dataframe, both overall and by row
    '''
    excluded_cols = ['healthCode', 'date', 'systolic', 'diastolic']
    selected_cols = list(set(df.columns) - set(excluded_cols))
    print('selected cols', selected_cols)
    
    # Exclude NaN values before checking for non-zero values
    valid_values = df[selected_cols].notnull() & df[selected_cols].ne(0)
    print('Number of non-zero and not NaN values:', valid_values.sum().sum())
    
    valid_rows = valid_values.any(axis=1)
    print('Number of rows with at least one non-zero and not NaN value:', valid_rows.sum())
    return valid_values.sum().sum()

def replace_nan(df):
    '''
    Replaces NaN values with 0s in rows where there is at least one non-NaN value
    '''
    # Identify rows with at least one non-NaN value
    mask = df.notnull().any(axis=1)
    
    # Replace NaN with 0 only in those rows
    df.loc[mask] = df.loc[mask].fillna(0)
    
    return df


def aggregate_dicts(dict1, dict2):
    '''
    Aggregates two dictionaries by averaging the values of common keys and adding the unique keys
    '''
    result = {}

    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        # Average the values if the key is in both dictionaries
        if key in dict1 and key in dict2:
            result[key] = (dict1[key] + dict2[key]) / 2
        elif key in dict1:
            result[key] = dict1[key]
        else:
            result[key] = dict2[key]

    # Sort the dictionary by value
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
    
    return result


def average_dicts(dict_list):
    '''
    Averages the values of a list of dictionaries with the same keys
    '''
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    # Iterate over all dictionaries and all keys and sum the values
    for d in dict_list:
        for k, v in d.items():
            sum_dict[k] += v
            count_dict[k] += 1

    # Divide the sums by the number of dictionaries that contain each key to get the averages
    avg_dict = {k: sum_dict[k] / count_dict[k] for k in sum_dict}

    return avg_dict


def log_exp(file, bp_predictor, aug='None', N=5, second_run=False, bootstrap=False, test_size=None, 
            historical=False, personalized=False):
    '''
    Logs the results of an experiment to a file
    '''

    # Extract the relevant information (metrics, model, parameters, etc.) from the bp_predictor object
    aug = aug
    dataset_size = test_size
    model = bp_predictor.model_type
    ntrees = bp_predictor.ntrees
    sys_mae = round(bp_predictor.mae['systolic'], 3)
    dias_mae = round(bp_predictor.mae['diastolic'], 3)
    top_N = list(bp_predictor.feature_importances.keys())[:N]   # Get only the keys of the top N features

    top_N = '; '.join(top_N)

    # Log the results as a new row in the file
    with open(file, 'a+') as f:
        # Checks that entry is not a duplicate row
        line = f'{aug},{dataset_size},{model},{ntrees},{sys_mae},{dias_mae},{top_N},{second_run},{bootstrap}\n'
        if line not in f.readlines():
            f.write(line)
        
    if personalized:
        top_N = 'N/A'
        
    print(f'''dataset size: {dataset_size[0]}, model: {model}, ntrees: {ntrees}, sys_mae: {sys_mae},
           dias_mae: {dias_mae}, top_n: {top_N}, second run: {second_run}, bootstrap: {bootstrap}, 
           historical: {historical}''')


def get_unique_healthCodes(dataset, threshold=2):
    grouped = dataset.dropna().groupby('healthCode').count()
    # Filter rows where count of unique healthCodes is greater than the threshold
    unique_healthCodes = dataset[dataset['healthCode'].isin(grouped[grouped['date'] > threshold].index)]
    unique_healthCodes = unique_healthCodes.dropna()['healthCode'].unique()
    return unique_healthCodes


def data_split(dataset, y_columns=['diastolic', 'systolic'], key_cols=['healthCode', 'date']):
    dataset = dataset.copy()

    # List of predictor columns (not key or target columns or historical BP columns)
    hist_bp = ['systolic_hist', 'diastolic_hist']
    cols = [col for col in dataset.columns if col not in key_cols + y_columns + hist_bp]
    # Drop rows where all values in the cols are NaN or 0
    dataset = dataset.dropna(subset=cols, how='all')
    # Drop rows where all values in the cols are 0
    dataset = dataset[(dataset[cols] != 0).any(axis=1)]


    train, test = train_test_split(dataset, test_size=0.2)
    y_train = train[y_columns]
    y_test = test[y_columns]
    x_train = train.drop(columns=y_columns, axis=1)
    x_test = test.drop(columns=y_columns, axis=1)

    # Saves the split to csv files
    train.to_csv('_data/train_test/train_nonstrat.csv', index=False)
    test.to_csv('_data/train_test/test_nonstrat.csv', index=False)

    return (x_train, y_train), (x_test, y_test)


def strat_data_split(dataset, y_columns=['diastolic', 'systolic'], key_cols=['healthCode']):
    '''
    Performs stratified data split based on the healthCode column
    '''
    dataset = dataset.copy()
    # List of predictor columns (not key or target columns)
    hist_bp = ['systolic_hist', 'diastolic_hist']
    cols = [col for col in dataset.columns if col not in key_cols + y_columns + hist_bp]
    # Drop rows where all values in the cols are NaN
    dataset = dataset.dropna(subset=cols, how='all')

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    for user in dataset[key_cols[0]].unique():
        user_data = dataset[dataset[key_cols[0]] == user]
        
        # Skip users with less than 2 data entries
        if user_data.shape[0] < 2:
            continue

        user_x_train, user_x_test, user_y_train, user_y_test = train_test_split(user_data.drop(y_columns, axis=1), 
                                                                                user_data[y_columns], test_size=0.2, 
                                                                                random_state=42)
        x_train = pd.concat([x_train, user_x_train])
        x_test = pd.concat([x_test, user_x_test])
        y_train = pd.concat([y_train, user_y_train])
        y_test = pd.concat([y_test, user_y_test])
    
    # Saves the split to csv files
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    train.to_csv('_data/train_test/train.csv', index=False)
    test.to_csv('_data/train_test/test.csv', index=False)
    

    return (x_train, y_train), (x_test, y_test)


def historical_BP(predictor, k):
    '''
    Here is where we list the historical BP values for each row in master_df
    '''
    data = predictor.copy()
    #Sorting the data by healthCode and date
    data.sort_values(by=['healthCode', 'date'], inplace=True)

    for col in ['systolic', 'diastolic']:
        data[col+'_hist'] = data.groupby('healthCode')[col].shift().transform(lambda x:x.ewm(span=k, adjust=True).mean())

    return data


def load_user_model(path):
    
    # Load the model
    loaded_model = XGBRegressor()
    loaded_model.load_model(path)
    
    return loaded_model

def load_json_as_dict(file_path):
    with open(file_path, 'r') as f:
        data_str = f.read()
    # Correct the format: replace single quotes with double quotes
    data_str = data_str.replace("'", '"')
    data_dict = json.loads(data_str)
    return data_dict