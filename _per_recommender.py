from _utils import load_user_model, load_json_as_dict
import os
from itertools import islice
import pandas as pd
import json
import sys

def get_per_recommendations(id, key, target, n=5, var_adjust=False, verbose=False, entry_num=None):

    # Define the paths to the model states, feature importances and the dataset
    model_path = 'personalized_model_states/model_states'
    f_imp_path = 'personalized_model_states/feature_importances'
    dataset_path = '_data/train_test'
    
    if entry_num:
        # Get the id of the user to make recommendations for    
        model_files = os.listdir(model_path)
        ids = [f.split('_')[0] for f in model_files]
        id = ids[entry_num]     # Extract the id of the user to make recommendations for

    # Load the models for the user
    model_sys = load_user_model(f'{model_path}/{id}_systolic.json')
    model_dia = load_user_model(f'{model_path}/{id}_diastolic.json')

    # Get the feature importances for the user
    file_path = f'{f_imp_path}/{id}.json'
    feature_importances = load_json_as_dict(file_path)
    top_n = dict(islice(feature_importances.items(), n))

    # Get predictor values for one of the user's test cases
    try:
        test_dataset = pd.read_csv(f'{dataset_path}/test.csv')
    except:
        test_dataset = pd.read_csv(f'../{dataset_path}/test.csv')
    # Order by date and get the most recent entry
    test_dataset['date'] = pd.to_datetime(test_dataset['date'])
    test_dataset = test_dataset.sort_values(by='date')
    test_entry = test_dataset[test_dataset['healthCode'] == id].iloc[[-1]]

    # Generate predictions for boths types of bp for the test entry
    expected_sys = 120.0
    expected_dia = 80.0
    x = test_entry.drop(key + target, axis=1)
    # print datatypes of all of x columns
    x = x.apply(pd.to_numeric, errors='coerce')

    sys_prediction = model_sys.predict(x)
    print(f'Predicted value: {sys_prediction}')
    sys_to_correct = sys_prediction - expected_sys
    dia_prediction = model_dia.predict(x)
    print(f'Predicted value: {dia_prediction}')
    dia_to_correct = dia_prediction - expected_dia

    if sys_to_correct < 0:
        sys_to_correct = 0
    if dia_to_correct < 0:
        dia_to_correct = 0

    total = expected_sys + expected_dia
    sys_w = expected_dia / total
    dia_w = expected_sys / total

    # Get weighted combination of systolic and diastolic to correct
    bp_to_correct = (sys_w * sys_to_correct + dia_w * dia_to_correct) / 2
    print('Weighted correction:', bp_to_correct)

    # If bp values are within healthy range, no correction is needed
    if bp_to_correct <= 0:
        print('No correction needed')
        return
    
    # Adjust the feature importances if the correction is too high or if explicitly specified
    if bp_to_correct >= 20.0:
        var_adjust = True
    if var_adjust:
        # summ all the top n feature importances values
        var_explained = sum(top_n.values())
        pred_adjustment = bp_to_correct / var_explained
        # Adjust the top n feature importances
        for key in top_n.keys():
            top_n[key] *= pred_adjustment

    # Multiply each top n prediction value by its corresponding feature importance
    recs = {}

    for key in top_n.keys():
        recs[key] = x[key].item() * top_n[key]

    if verbose:
        print('\n Recommendations:')
        for key in top_n.keys():
            print(f'Activity: {key}  -   Value: {x[key].item()}   -  imp_score: {top_n[key]}' 
                  f'-  Rec: {recs[key]}')
        try:
            train_dataset = pd.read_csv(f'{dataset_path}/train.csv')
        except:
            train_dataset = pd.read_csv(f'../{dataset_path}/train.csv')
        target_user_entries = train_dataset[train_dataset['healthCode'] == id]
        print('\n Target user training entries:')
        print(target_user_entries)
    
    return recs


if __name__ == '__main__':
    user_config = 'configs/recommendation/per-user_config.json'    # default path to the user config
    query_config = 'configs/recommendation/query_config.json'      # default path to the query config

    # If arguments for the user and/or query configs are provided, use them to replace the default paths
    if len(sys.argv) > 1:
        user_config = sys.argv[1]       # argument path to the user config
    if len(sys.argv) > 2:
        query_config = sys.argv[2]      # argument path to the query config

    # Load the id (healthCode) the user to make recommendations for from the user config
    with open(user_config, 'r') as f:
        config = json.load(f)
    id = config['healthCode']

    # Load the query parameters from the query config
    with open(query_config, 'r') as f:
        query_config = json.load(f)
    key = query_config['key']                   # names of the key columns (healthCode, date)
    target = query_config['target']             # name of the columns to predict (systolic, diastolic)
    n = query_config['n']                       # how many features to provide recommendations for
    var_adjust = query_config['var_adjust']     # whether to adjust the feature importances based on the variance explained
    verbose = query_config['verbose']           # whether to print the recommendations to the console

    recs = get_per_recommendations(id, key, target, n=n, var_adjust=var_adjust, verbose=verbose)