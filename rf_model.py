from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import aggregate_dicts


def data_split(df, y_columns=['diastolic', 'systolic']):
  train, test = train_test_split(df, test_size=0.2)
  y_train = train[y_columns]
  y_test = test[y_columns]
  x_train = train.drop(y_columns, axis=1)
  x_test = test.drop(y_columns, axis=1)
  return (x_train, y_train), (x_test, y_test)


def rfr_predict(dataset, ntrees=30):
  dataset = dataset.copy()
  angel_dataset = dataset.drop(['healthCode', 'date'], axis=1)
  angel_dataset = angel_dataset.dropna()
  print('Shape complete dataset with no NaN values: ' , angel_dataset.shape)
  (x_train, y_train), (x_test, y_test) = data_split(angel_dataset)

  model = RandomForestRegressor(n_estimators=ntrees)
  feature_importances = []
  mse = []
  mae = []

  # Predicts and calculates metrics for systolic and diastolic blood pressure separately
  for bp_type in ['systolic', 'diastolic']:
    model = model.fit(x_train, y_train[bp_type])
    pred = model.predict(x_test).round()
    mse.append(mean_squared_error(pred,y_test[bp_type]))
    mae.append(mean_absolute_error(pred,y_test[bp_type]))
    feature_importances.append(get_feature_importances(model, x_train.columns))
  
  # Aggregates systolic and diastolic prediction feature importances
  feature_importances = aggregate_dicts(feature_importances[0], feature_importances[1]) 
  
  return mse, mae, feature_importances


def get_feature_importances(model, features):
  importances = model.feature_importances_
  feature_importances = {name: score for name, score in zip(features, importances)}
  feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)}
  return feature_importances
  

  
