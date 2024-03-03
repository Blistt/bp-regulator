from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import aggregate_dicts, average_dicts
from collections import defaultdict
from sklearn.utils import resample


class BloodPresurePredictor:
    def __init__(self, model_type='rf', ntrees=30):
        self.model_type = model_type
        self.ntrees = ntrees
        self.model = defaultdict()
        self.pred = defaultdict()
        self.mse = defaultdict()
        self.mae = defaultdict()
        self.feature_importances = None
        self.dataset_size = None
      

    def data_split(self, dataset, y_columns=['diastolic', 'systolic']):
        train, test = train_test_split(dataset, test_size=0.2)
        y_train = train[y_columns]
        y_test = test[y_columns]
        x_train = train.drop(columns=y_columns, axis=1)
        x_test = test.drop(columns=y_columns, axis=1)
        return (x_train, y_train), (x_test, y_test)
    

    def get_feature_importances(self, features, bp_type):
        importances = self.model[bp_type].feature_importances_
        feature_importances = {feature: score for feature, score in zip(features, importances)}
        feature_importances = {k: v for k, v in sorted(feature_importances.items(),   # Sorts the dictionary by value
                              key=lambda item: item[1], reverse=True)}
        if 'heart_rate' in feature_importances:
            del feature_importances['heart_rate']
        return feature_importances
    

    def predict(self, dataset):
        # Pre-processes the dataset
        dataset = dataset.copy()
        dataset = dataset.drop(columns=['healthCode', 'date'], axis=1)
        dataset = dataset.dropna()
        self.dataset_size = dataset.shape[0]  # Stores the size of the valid dataset
        (x_train, y_train), (x_test, y_test) = self.data_split(dataset)

        # Initializes the model and lists to store metrics and feature importances
        if self.model_type == 'rf':
            model = RandomForestRegressor(n_estimators=self.ntrees)
        elif self.model_type == 'xgb':
            model = XGBRegressor(n_estimators=self.ntrees)
        
        temp_f_importances = []
        for bp_type in ['systolic', 'diastolic']:
            self.model[bp_type] = model.fit(x_train, y_train[bp_type])
            self.pred[bp_type] = model.predict(x_test).round()
            self.mse[bp_type] = mean_squared_error(y_test[bp_type], self.pred[bp_type])
            self.mae[bp_type] = mean_absolute_error(y_test[bp_type], self.pred[bp_type])
            temp_f_importances.append(self.get_feature_importances(x_train.columns, bp_type))
        self.feature_importances = aggregate_dicts(temp_f_importances[0], temp_f_importances[1])


    def bootstrap(self, dataset, iterations, size):
        bootstrap_size = int(size * dataset.shape[0])
        mse_values = defaultdict(list)
        mae_values = defaultdict(list)
        pred = defaultdict(list)
        feature_importances = []

        for i in range(iterations):
            # Resamples the dataset
            resampled_dataset = resample(dataset, n_samples=bootstrap_size)
            self.predict(resampled_dataset)
            for bp_type in ['systolic', 'diastolic']:
                mse_values[bp_type].append(self.mse[bp_type])
                mae_values[bp_type].append(self.mae[bp_type])
                pred[bp_type].append(self.pred[bp_type])
            feature_importances.append(self.feature_importances)

        # Averages the metrics for each blood pressure type
        for bp_type in ['systolic', 'diastolic']:
            self.mse[bp_type] = sum(mse_values[bp_type]) / iterations
            self.mae[bp_type] = sum(mae_values[bp_type]) / iterations
            self.pred[bp_type] = None

        # Averages the feature importances for all iterations
        self.feature_importances = average_dicts(feature_importances)





