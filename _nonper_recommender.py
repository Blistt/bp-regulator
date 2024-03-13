from _utils import load_user_model, load_json_as_dict
from itertools import islice
import pandas as pd
import json
import sys

def get_nonper_recommendations(entry, key, target, n=5, var_adjust=False, verbose=False):

    # Define the paths to the model states, feature importances and the dataset
    model_path = 'nonpersonalized_model_states/model_states'
    f_imp_path = 'nonpersonalized_model_states/feature_importances'

    # Load the models for the user
    model_sys = load_user_model(f'{model_path}/allusers_systolic.json')
    model_dia = load_user_model(f'{model_path}/allusers_diastolic.json')

    # Get the feature importances for the user
    file_path = f'{f_imp_path}/all_users.json'
    feature_importances = load_json_as_dict(file_path)
    top_n = dict(islice(feature_importances.items(), n))

    # Generate predictions for boths types of bp for the test entry
    sys_prediction = model_sys.predict(entry)
    print(f'Predicted value: {sys_prediction}')
    dia_prediction = model_dia.predict(entry)
    print(f'Predicted value: {dia_prediction}')
    
    # Compute correction needed to get the BP predictions in the healthy range
    expected_sys = 120.0
    expected_dia = 80.0
    sys_to_correct = sys_prediction - expected_sys
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
        recs[key] = entry[key].item() * top_n[key]

    if verbose:
        print('\n Recommendations:')
        for key in top_n.keys():
            print(f'Activity: {key}  -   Value: {entry[key].item()}   -  imp_score: {top_n[key]}' 
                  f'-  Rec: {recs[key]}')
    
    return recs


if __name__ == '__main__':
    user_config = 'configs/recommendation/nonper-user_config.json'    # default path to the user config
    query_config = 'configs/recommendation/query_config.json'      # default path to the query config

    # If arguments for the user and/or query configs are provided, use them to replace the default paths
    if len(sys.argv) > 1:
        user_config = sys.argv[1]       # path to the user config
    if len(sys.argv) > 2:
        query_config = sys.argv[2]      # path to the query config

    # Load the user provided predictor values from the user config
    with open(user_config, 'r') as f:
        user_config = json.load(f)
    entry_df = pd.DataFrame([user_config])

    # Load the query parameters from the query config
    with open(query_config, 'r') as f:
        query_config = json.load(f)
    key = query_config['key']                   # names of the key columns (healthCode, date)
    target = query_config['target']             # name of the columns to predict (systolic, diastolic)
    n = query_config['n']                       # how many features to provide recommendations for
    var_adjust = query_config['var_adjust']     # whether to adjust the feature importances based on the variance explained
    verbose = query_config['verbose']           # whether to print the recommendations to the console

    get_nonper_recommendations(entry_df, key, target, n=n, var_adjust=var_adjust, verbose=verbose)