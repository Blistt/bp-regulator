from utils import load_user_model, load_json_as_dict
import os
from itertools import islice
import pandas as pd
import numpy as np

def get_nonper_recommendations(x, key, target, n=5, var_adjust=False, verbose=False):

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
    expected_sys = 120.0
    expected_dia = 80.0

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
        recs[key] = x[key] * top_n[key]

    if verbose:
        print('\n Recommendations:')
        for key in top_n.keys():
            print(f'Activity: {key}  -   Value: {x[key].item()}   -  imp_score: {top_n[key]}' 
                  f'-  Rec: {recs[key].item()}')
    
    return recs