from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def data_split(df, y_columns=['diastolic', 'systolic']):
  train, test = train_test_split(df, test_size=0.2)
  y_train = train[y_columns]
  y_test = test[y_columns]
  x_train = train.drop(y_columns, axis=1)
  x_test = test.drop(y_columns, axis=1)
  return (x_train, y_train), (x_test, y_test)

def rfr_predict(dataset, target_col, ntrees=30):
  dataset = dataset.copy()
  angel_dataset = dataset.drop(['healthCode', 'date'], axis=1)
  angel_dataset = angel_dataset.dropna()
  print('Shape complete dataset with no NaN values: ' , angel_dataset.shape)
  (x_train, y_train), (x_test, y_test) = data_split(angel_dataset)
  model = RandomForestRegressor(n_estimators=ntrees)
  model = model.fit(x_train, y_train[target_col])
  pred = model.predict(x_test).round()
  mse = mean_squared_error(pred,y_test[target_col])
  mae = mean_absolute_error(pred,y_test[target_col])
  return (mse, mae)
