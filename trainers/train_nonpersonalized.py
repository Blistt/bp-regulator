from _bp_predictor import BloodPresurePredictor
from _utils import log_exp, data_split, historical_BP
import os

def train_nonpersonazlied(dataset, model, ntrees, N, key, target, log_path='', bootstrap=False, 
               bootstrap_size=0.8, aug='None', historical=True, second_run=True):
    # Add historical blood pressure to the dataset if specified
    if historical:
        dataset = historical_BP(dataset, 3)

    # Split dataset into train and test sets of features and labels
    (x_train, y_train), (x_test, y_test) = data_split(dataset, y_columns=target, key_cols=key)
    x_train = x_train.drop(key, axis=1)
    x_test = x_test.drop(key, axis=1)

    # Train & evaluate model
    bp_predictor = BloodPresurePredictor(model, ntrees)             # Create model
    bp_predictor.fit(x_train, y_train, bootstrap, bootstrap_size)   # Train model
    bp_predictor.evaluate(x_test, y_test)                           # Evaluate the model

    # Save model and log results
    path_state = 'nonpersonalized_model_states/model_states'
    path_feat = 'nonpersonalized_model_states/feature_importances'
    if not os.path.exists(path_state):
        os.makedirs(path_state)
    if not os.path.exists(path_feat):
        os.makedirs(path_feat)
    bp_predictor.model['systolic'].save_model(f'{path_state}/allusers_systolic.json')
    bp_predictor.model['diastolic'].save_model(f'{path_state}/allusers_diastolic.json')
    # Save dict of feature importances
    with open(f'{path_feat}/all_users.json', 'w') as f:
        f.write(str(bp_predictor.feature_importances))
    # log results
    log_exp(log_path, bp_predictor, aug=aug, N=N, second_run=False, bootstrap=bootstrap, test_size=x_test.shape,
            historical=historical, personalized=False)     

    if second_run:
        # Second run with top N features
        top_n = list(bp_predictor.feature_importances.keys())[:N]        # get top N features from prior run
        x_train = x_train[top_n]                                         # select top N features
        bp_predictor.fit(x_train[top_n], y_train)                        # predict with top N features
        bp_predictor.evaluate(x_test[top_n], y_test)                     # evaluate with top N features
        
        # Save model and log results
        bp_predictor.model['systolic'].save_model(f'{path_state}/allusers_systolic.json')
        bp_predictor.model['diastolic'].save_model(f'{path_feat}/allusers_diastolic.json')
        # Save dict of feature importances
        with open(f'{path_feat}/all_users.json', 'w') as f:
            f.write(str(bp_predictor.feature_importances))
        # log results
        log_exp(log_path, bp_predictor, aug=aug, N=N, second_run=True, bootstrap=bootstrap, test_size=x_test.shape,
                historical=historical, personalized=False)     