from _bp_predictor import BloodPresurePredictor
from _utils import log_exp, get_unique_healthCodes, average_dicts, strat_data_split, historical_BP
from collections import defaultdict
import os

def train_personalized(dataset, model, ntrees, N, key, target, log_path='', bootstrap=False, 
               bootstrap_size=0.8, aug='None', second_run=False, historical=True):
    # Add historical blood pressure to the dataset if specified
    if historical:
        dataset = historical_BP(dataset, 3)
    
    # Split dataset into train and test sets of features and labels
    (x_train, y_train), (x_test, y_test) = strat_data_split(dataset, y_columns=target, key_cols=key)
    x_train_keys = x_train[key]
    x_test_keys = x_test[key]
    x_train = x_train.drop(key, axis=1)
    x_test = x_test.drop(key, axis=1)

    # Get baseline model by running on all users
    bp_predictor = BloodPresurePredictor(model, ntrees)
    bp_predictor.fit(x_train, y_train, bootstrap, bootstrap_size)

    # Get all unique healthCodes
    all_users = get_unique_healthCodes(dataset)

    # Initialize lists to store metrics results
    mae = defaultdict(list)
    mse = defaultdict(list)
    temp_feature_importances = []
    # Personalize the model for each user
    for user in all_users:
        tr_mask = x_train_keys.iloc[:, 0] == user
        test_mask = x_test_keys.iloc[:, 0] == user
        x_train_user, y_train_user = x_train[tr_mask], y_train[tr_mask]

        x_test_user, y_test_user = x_test[test_mask], y_test[test_mask]

        # Skips if there are no samples for the user
        if x_train_user.shape[0] < 1 or x_test_user.shape[0] < 1:
            continue

        else:
            bp_predictor.fine_tune(x_train_user, y_train_user)                   # Fit the personalized model
            bp_predictor.evaluate(x_test_user, y_test_user, fine_tuned=True)     # Evaluate the personalized model
            # Performs second run with top N features if specified
            if second_run:
                top_n = list(bp_predictor.feature_importances.keys())[:N]
                bp_predictor.fine_tune(x_train_user[top_n], y_train_user)
                bp_predictor.evaluate(x_test_user[top_n], y_test_user, fine_tuned=True)
            for bp_type in target:
                mae[bp_type].append(bp_predictor.mae[bp_type])
                mse[bp_type].append(bp_predictor.mse[bp_type])
            temp_feature_importances.append(bp_predictor.feature_importances)

            # Saves the model and the feature importances for each user
            path_state = 'personalized_model_states/model_states'
            path_feat = 'personalized_model_states/feature_importances'
            if not os.path.exists(path_state):
                os.makedirs(path_state)
            if not os.path.exists(path_feat):
                os.makedirs(path_feat)
            for bp_type in target:
                bp_predictor.ftmodel[bp_type].save_model(f'{path_state}/{user}_{bp_type}.json')
            # Save dict of feature importances
            with open(f'{path_feat}/{user}.json', 'w') as f:
                f.write(str(bp_predictor.feature_importances))

    if len(mae) == 0:
        print('No testing samples')
        return
    
    # Average metrics for all users
    for bp_type in target:
        bp_predictor.mae[bp_type] = sum(mae[bp_type]) / len(mae[bp_type])
        bp_predictor.mse[bp_type] = sum(mse[bp_type]) / len(mse[bp_type])
    bp_predictor.feature_importances = average_dicts(temp_feature_importances)

    # log results
    log_exp(log_path, bp_predictor, aug=aug, N=N, second_run=second_run, bootstrap=bootstrap, 
            test_size=x_test.shape, historical=historical, personalized=True)