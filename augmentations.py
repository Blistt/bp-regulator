import pandas as pd

def rolling_k_days(predictor, k):
  '''
  Populates missing values a time series table with the rolling average of the k prior days
  (not all days will be populated, as any day for which there is no data in the previous k days will
  remain as an empty value)
  '''
  predictor_df = predictor.copy()
  print('original shape', predictor_df.shape)

  # Resample to daily data (create an entry for everyday in the dates range, even if it has empty values)
  predictor_df['date'] = pd.to_datetime(predictor_df['date'])
  predictor_df.set_index('date', inplace=True)
  # predictor_df = predictor_df.groupby('healthCode').resample('D').mean()
  print('post sample shape', predictor_df.shape)

  # Fill in missing days with NaNs
  predictor_df = predictor_df.reset_index().set_index('date').groupby('healthCode', group_keys=False).apply(lambda x: x.asfreq('D')).reset_index()

  predictor_df = predictor_df.sort_values(['healthCode', 'date'])

  # Select variables (columns) to augment
  cols_to_augment = predictor_df.columns[:2]  # augment all but the first two
  cols_to_augment = ['floors']

  # Calculate rolling average of k days to populate as many days with missing data as possible
  predictor_df[cols_to_augment] = predictor_df.groupby('healthCode')[cols_to_augment].rolling(window=k, min_periods=1).mean().reset_index(0, drop=True)

  return predictor_df