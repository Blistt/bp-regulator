import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.neighbors import NearestNeighbors

def rolling_k_days(predictor, k):
    '''
    Populates missing values a time series table with the rolling average of the k prior days
    (not all days will be populated, as any day for which there is no data in the previous k days will
    remain as an empty value)
    '''

    # Ensure your data is sorted by 'date'
    predictor.sort_values(by=['healthCode', 'date'], inplace=True)

    # Get all column names except 'healthCode' and 'date'
    cols = [col for col in predictor.columns if col not in ['healthCode', 'date', 'systoclic', 'diastolic']]

    # Apply the operation to all other columns
    for col in cols:
        predictor[col] = predictor.groupby('healthCode')[col].transform(lambda x: x.rolling(window=k, min_periods=1).mean())
        predictor[col] = predictor.groupby('healthCode')[col].ffill()  

    return predictor


# def knn_impute(predictor):
#     cols = [col for col in predictor.columns if col not in ['healthCode', 'date', 'systolic', 'diastolic']]

#     # Create the imputer
#     imputer = KNNImputer(n_neighbors=3)

#     # Split the DataFrame by 'id', apply the imputation, and concatenate the results
#     predictor_imputed = pd.concat(
#         (pd.DataFrame(imputer.fit_transform(sub_df[cols]), columns=cols) 
#          for id, sub_df in predictor.groupby('healthCode')),
#         ignore_index=True
#     )
#     # Convert the result back t+o a DataFrame (if necessary)
#     predictor_imputed = pd.DataFrame(predictor_imputed, columns=predictor.columns)
#     print(predictor.head())
#     print(predictor_imputed.head())

#     return predictor_imputed


def get_neighbors(df, index, k=3):
    '''
    Used only for data exploration purposes. This function returns the k nearest neighbors 
    of the specified index in the DataFrame
    '''
    # Select the columns & index to use for the nearest neighbors
    df.fillna(0, inplace=True)
    cols = [col for col in df.columns if col not in ['healthCode', 'date', 'systoclic', 'diastolic']]
    X = df[df['healthCode'] == df['healthCode'].iloc[index]]
    X = X[cols]
    X = X.fillna(0)

    # Ensures that we are not asking for more neighbors than there are entries with selected healthCode
    print('entries with selected healthCode', X.shape)
    if k > X.shape[0]:
        k = X.shape[0] - 1

    # Reset the index and keep the old index
    X.reset_index(inplace=True)

    # Create the estimator
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X[cols])

    # Get the k nearest neighbors of the specified index
    distances, indices = nbrs.kneighbors(X[X['index'] == index][cols].values.reshape(1, -1))

    print(df.iloc[X.iloc[indices[0]]['index']])
