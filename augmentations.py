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
    predictor = predictor.copy()
    # Ensure the data is sorted by 'date'
    predictor.sort_values(by=['healthCode', 'date'], inplace=True)
    # Replace 0s with NaNs
    # predictor.replace(0, np.nan, inplace=True)

    # Get all column names except the indexing and target variable columns
    cols = [col for col in predictor.columns if col not in ['healthCode', 'date', 'systolic', 'diastolic']]

    # Apply the operation to all other columns
    for col in cols:
        predictor_rolling_mean = predictor.groupby('healthCode')[col].transform(lambda x: x.rolling(window=k, min_periods=1).mean())
        predictor[col] = predictor.groupby('healthCode')[col].fillna(predictor_rolling_mean)

    return predictor


def check_columns(predictor):
    '''
    Checks which columns contain at least one non-missing value in each group.
    '''
    cols = [col for col in predictor.columns if col not in ['healthCode', 'date', 'systolic', 'diastolic']]

    # Initialize an empty dictionary to store the results
    columns_with_data = {}

    # Iterate over each group and check which columns contain at least one non-missing value
    for id, sub_df in predictor.groupby('healthCode'):
        columns_with_data[id] = sub_df[cols].replace(-1, np.nan).dropna(axis=1, how='all').columns.tolist()

    return columns_with_data


def knn_impute_inter_user(predictor, k=3):
    '''
    Populates missing values using a KNN imputation method 
    * Considers the k nearest neighbors across all users
    '''
    predictor = predictor.copy()
    predictor.fillna(-1, inplace=True)   # Replace NaN with -1
    bad_cols = ['healthCode', 'date', 'systolic', 'diastolic']      # Potential cols to remove
    remove = [el for el in bad_cols if el in predictor.columns]     # Cols to remove
    cols = [col for col in predictor.columns if col not in remove]  # Cols to impute

    # Create the imputer
    imputer = KNNImputer(n_neighbors=k, missing_values=-1, weights='distance')

    # Apply the imputation to the entire DataFrame at once
    predictor_imputed = pd.DataFrame(imputer.fit_transform(predictor[cols]), columns=cols)

    # Add all the other columns back    
    predictor_imputed = pd.concat([predictor[remove],
                                    predictor_imputed], axis=1)
    predictor_imputed = predictor_imputed.replace(-1, np.nan)
    return predictor_imputed


def knn_impute_intra_user(predictor, k=3):
    '''
    Populates missing values in a time series table using a KNN imputation method
    * Only considers the k nearest neighbors within the same user
    '''
    predictor = predictor.copy()
    predictor.fillna(-1, inplace=True)   # Replace NaN with -1
    bad_cols = ['healthCode', 'date', 'systolic', 'diastolic']
    remove = [el for el in bad_cols if el in predictor.columns]
    cols = [col for col in predictor.columns if col not in remove]

    # Create the imputer
    imputer = KNNImputer(n_neighbors=k, missing_values=-1, weights='distance')

    # Get the columns with data for each group
    columns_with_data = check_columns(predictor)

    # Iterate over each group, apply the imputation, and merge the results back into the original DataFrame
    for id, sub_df in predictor.groupby('healthCode'):
        # Get the columns with data for the current group
        cols_with_data = columns_with_data[id]
        # Apply the imputation to these columns
        imputed_values = imputer.fit_transform(sub_df[cols_with_data])
        # Replace the values in the original DataFrame with the imputed values
        predictor.loc[sub_df.index, cols_with_data] = imputed_values
    
    predictor = predictor.replace(-1, np.nan)

    return predictor

def get_neighbors(df, index, k=3):
    '''
    Used only for data exploration purposes. This function returns the k nearest neighbors 
    of the specified index in the DataFrame
    '''
    df = df.copy()
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
