from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from _utils import aggregate_dicts, average_dicts
from collections import defaultdict
from sklearn.utils import resample
from _utils import data_split
import pandas as pd


class BloodPresurePredictor:
    def __init__(self, model_type='rf', ntrees=30, target_cols=['systolic', 'diastolic']):
        self.model_type = model_type
        self.ntrees = ntrees
        self.model = defaultdict()
        self.ftmodel = defaultdict()
        self.pred = defaultdict()
        self.mse = defaultdict()
        self.mae = defaultdict()
        self.feature_importances = None
        self.target_cols = target_cols


    def get_feature_importances(self, features, bp_type, ft=True):
        '''
        Extracts the feature importances from the model and stores them in a dictionary
        '''
        if ft:
            importances = self.ftmodel[bp_type].feature_importances_
        else:
            importances = self.model[bp_type].feature_importances_
        feature_importances = {feature: score for feature, score in zip(features, importances)}
        feature_importances = {k: v for k, v in sorted(feature_importances.items(),   # Sorts the dictionary by value
                              key=lambda item: item[1], reverse=True)}
        
        # Remove features that do not constitute behavioral changes
        if 'heart_rate' in feature_importances:
            del feature_importances['heart_rate']
        if 'systolic_hist' in feature_importances:
            del feature_importances['systolic_hist']
        if 'diastolic_hist' in feature_importances:
            del feature_importances['diastolic_hist']

        return feature_importances
    

    def fit(self, x_train, y_train, bootstrap=False, bootstrap_size=0.5):
        '''
        Fits the selected model (Random Forest or XGBoost) to the training data and stores
        the feature importances in a dictionary
        '''
        temp_f_importances = []
        # Creates a model for each blood pressure type
        for bp_type in self.target_cols:
            # Initializes the model
            if self.model_type == 'rf':
                model = RandomForestRegressor(n_estimators=self.ntrees, bootstrap=bootstrap, max_samples=bootstrap_size)
            elif self.model_type == 'xgb':
                model = XGBRegressor(n_estimators=self.ntrees, subsample=bootstrap_size)
            self.model[bp_type] = model.fit(x_train, y_train[bp_type])
            temp_f_importances.append(self.get_feature_importances(x_train.columns, bp_type, ft=False))
        self.feature_importances = aggregate_dicts(temp_f_importances[0], temp_f_importances[1])


    def evaluate(self, x_test, y_test, fine_tuned=False):
        '''
        Evaluates the model on the test data and stores the predictions and metrics
        '''
        for bp_type in self.target_cols:
            if fine_tuned:
                self.pred[bp_type] = self.ftmodel[bp_type].predict(x_test).round()
            else:
                self.pred[bp_type] = self.model[bp_type].predict(x_test).round()
            self.mse[bp_type] = mean_squared_error(y_test[bp_type], self.pred[bp_type])
            self.mae[bp_type] = mean_absolute_error(y_test[bp_type], self.pred[bp_type])


    def fine_tune(self, x_train, y_train, bootstrap_size=1.0):
        '''
        Fine tunes the model trained on all users to personalize the model for each user
        '''
        # Ensures that the model is not a Random Forest model, but an XGBoost model
        if self.model_type == 'rf':
            print('Cannot fine tune a Random Forest model')
            return

        temp_f_importances = []
        for bp_type in self.target_cols:
            # Use the base model and continue training on the user's data
            self.ftmodel[bp_type] = XGBRegressor(n_estimators=self.ntrees, subsample=bootstrap_size)
            self.ftmodel[bp_type].fit(x_train, y_train[bp_type], xgb_model=self.model[bp_type])
            temp_f_importances.append(self.get_feature_importances(x_train.columns, bp_type, ft=True))
        self.feature_importances = aggregate_dicts(temp_f_importances[0], temp_f_importances[1])




