import pandas as pd
import numpy as np
from collections import defaultdict

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
    avg_dict = defaultdict(float)

    # Iterate over all dictionaries and all keys and sum the values
    for d in dict_list:
        for k, v in d.items():
            avg_dict[k] += v

    # Divide the sums by the number of dictionaries to get the averages
    for k in avg_dict:
        avg_dict[k] /= len(dict_list)

    return avg_dict


def log_exp(file, bp_predictor, aug='None', N=5, double=False, bootstrap=False):
    '''
    Logs the results of an experiment to a file
    '''

    # Extract the relevant information (metrics, model, parameters, etc.) from the bp_predictor object
    aug = aug
    dataset_size = bp_predictor.dataset_size
    model = bp_predictor.model_type
    ntrees = bp_predictor.ntrees
    sys_mae = round(bp_predictor.mae['systolic'], 3)
    dias_mae = round(bp_predictor.mae['diastolic'], 3)
    top_N = list(bp_predictor.feature_importances.keys())[:N]   # Get only the keys of the top N features

    top_N = '; '.join(top_N)

    # Log the results as a new row in the file
    with open(file, 'a+') as f:
        # Checks that entry is not a duplicate row
        line = f'{aug},{dataset_size},{model},{ntrees},{sys_mae},{dias_mae},{top_N},{double},{bootstrap}\n'
        if line not in f.readlines():
            f.write(line)
    
    print(f'''aug:, {aug}, dataset size: {dataset_size}, model: {model}, ntrees: {ntrees}, sys_mae: {sys_mae}, dias_mae: {dias_mae}, 
          top_n: {top_N}, double run: {double}, bootstrap: {bootstrap}''')


