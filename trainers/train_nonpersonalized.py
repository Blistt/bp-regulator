import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd())) if str(Path.cwd()) not in sys.path else None
from _bp_predictor import BloodPresurePredictor
from _utils import log_exp, data_split, historical_BP
import os
import pandas as pd
import json


def train_nonpersonazlied(datapath, model, ntrees, n, key, target, exclude=[], log_path='', bootstrap=False, 
               bootstrap_size=0.8, aug='baseline', historical=True, second_run=True):
    
    dataset = pd.read_csv(f'{datapath}/{aug}.csv')      # Load dataset with specified augmentations
    # Add historical blood pressure to the dataset if specified
    if historical:
        dataset = historical_BP(dataset, 3)

    # Split dataset into train and test sets of features and labels
    (x_train, y_train), (x_test, y_test) = data_split(dataset, y_columns=target, key_cols=key, datapath=datapath)
    x_train = x_train.drop(key, axis=1)
    x_test = x_test.drop(key, axis=1)

    # Train & evaluate model
    bp_predictor = BloodPresurePredictor(model, ntrees, target_cols=target, exlude_cols=exclude)  # Create model
    bp_predictor.fit(x_train, y_train, bootstrap, bootstrap_size)   # Train model
    bp_predictor.evaluate(x_test, y_test)                           # Evaluate the model

    # Save model and log results
    dir_corr = Path.cwd()
    path_state = dir_corr/'nonpersonalized_model_states'/'model_states'
    path_feat = dir_corr/'nonpersonalized_model_states'/'feature_importances'
    if not os.path.exists(path_state):
        os.makedirs(path_state)
    if not os.path.exists(path_feat):
        os.makedirs(path_feat)
    for bp_type in bp_predictor.target_cols:
        bp_predictor.model[bp_type].save_model(f'{path_state}/allusers_{bp_type}.json')

    # Save dict of feature importances
    with open(f'{path_feat}/all_users.json', 'w') as f:
        f.write(str(bp_predictor.feature_importances))
    # log results
    log_exp(log_path, bp_predictor, aug=aug, n=n, second_run=False, bootstrap=bootstrap, test_size=x_test.shape,
            historical=historical, personalized=False)     

    if second_run:
        # Second run with top N features
        top_n = list(bp_predictor.feature_importances.keys())[:n]        # get top N features from prior run
        x_train = x_train[top_n]                                         # select top N features
        bp_predictor.fit(x_train[top_n], y_train)                        # predict with top N features
        bp_predictor.evaluate(x_test[top_n], y_test)                     # evaluate with top N features
        
        # Save model and log results
        for bp_type in bp_predictor.target_cols:
            bp_predictor.model[bp_type].save_model(f'{path_state}/allusers_{bp_type}.json')
        # Save dict of feature importances
        with open(f'{path_feat}/all_users.json', 'w') as f:
            f.write(str(bp_predictor.feature_importances))
        # log results
        log_exp(log_path, bp_predictor, aug=aug, n=n, second_run=True, bootstrap=bootstrap, test_size=x_test.shape,
                historical=historical, personalized=False)     
        

if __name__ == '__main__':
    dir_corr = Path.cwd()
    model_config = dir_corr/'configs'/'training'/'nonper-model_config.json'     # default path to the model config
    dataset_config = dir_corr/'configs'/'training'/'dataset_config.json'         # default path to the dataset config

    ############################### MODEL PARAMETERS ###############################
    with open(model_config) as f:
        model_config = json.load(f)
    n = model_config['n']                               # Number of most important features to display
    model = model_config['model']                       # rf or xgb (Random Forest or XGBoost)
    ntrees = model_config['ntrees']                     # Number of trees in the forest
    second_run = model_config['second_run']             # Whether to use a second run with top N features or not
    bootstrap = model_config['bootstrap']               # Whether to use bootstrap samples
    bootstrap_size = model_config['bootstrap_size']     # Portion of the dataset to sample for bootstrap
    historical = model_config['historical']             # Whether to use historical BP or not

    ############################### DATASET PARAMETERS ###############################
    with open(dataset_config) as f:
        dataset_config = json.load(f)
    datapath = dataset_config['datapath']  # Path of the dataset
    aug = dataset_config['aug']            # Type of augmentation to use
    key = dataset_config['key']            # Columns to use as key
    target = dataset_config['target']      # Columns to predict
    exclude = dataset_config['exclude']    # Columns to exclude from the feature importance analysis
    log_path = dataset_config['log_path']  # Path of file to log experiment results

    train_nonpersonazlied(datapath, model, ntrees, n, key, target, log_path=log_path, second_run=second_run, 
           bootstrap=bootstrap, bootstrap_size=bootstrap_size, aug=aug, historical=historical, exclude=exclude)
