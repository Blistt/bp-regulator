# xgb-blood-pressure-tracker

This repository implements a system that recommends specific behavioral changes of daily activity to a given user in order to regulate daily blood pressure levels.

The recommendations are obtained by using an ensemble of gradient-boosted trees (XGBoost) to predict daily blood pressure (BP) levels from a set of biometric markers measured by a smartwach (e.g., sleep minutes, steps, calories burned). Feature importance scores (\% of variance in the output explained by the variable) are obtained for each biometric predictor variable. Recommendations are then presented as concrete quantities (e.g., 25 more sleep minutes, 500 more steps) derived from the feature scores and the predicted BP values. 

## Requirements 
### (conda installation)
Run the following command:

```bash
conda create --name <env> --file requirements.txt
```

Replace env with the desired name for the virtual environment

### (Manual installation)
- xgboost == 1.7.3 
- scikit-learn = 1.2.2
- python == 3.11.7 
- numpy == 1.26.3 
- pandas == 2.1.4

Miscellaneous
- tqdm == 4.65.0

## Get recommendations
This system supports two types of recommendations: 1) personalized recommendations, 2) non-personalized recommendations.  

### Non-personalized recommendation
Non-personalized recommendations will use the same model for all users. Ideal for users not in the database yet, with no historical biometric and BP data provided yet. 
To get non-personalized recommendations, go to the `nonper_user_config.json` file in the `configs/recommendation` directory to enter the predictor variable values (e.g., sleep minutes, steps, historical_bp). Do the same with the `query_config.json` file located in the same directory for query parameters (e.g., how many features to recommend for).

Run the following command:
```bash
python _nonper_recommender.py
```

### Personalized recommendation
Personalized recommendations will use a distinct, fine-tuned model for each user. Meant for users already in the database, with a history of biometric and BP data on which a personalized model was fine tuned. To get personalized recommendations, go to the `per_user_config.json` file in the `configs/recommendation` directory to enter the ID of the user the recommendations are for. The system will retrieve the most recent predictor variable values from the database. Do the same with the `query_config.json` file located in the same directory for query parameters (e.g., how many features to recommend for).

Run the following command:
```bash
python _per_recommender.py
```

## Training
This system employs one of two trainable configurations: 1) non-personalized training, or 2) personalized training. 
### Non-personalized training
Trains a single model in the prediction of the daily BP values of the entire database of users. To train a non-personalized model, run the following command:
```bash
python trainers/train_nonpersonalized.py
```
Make sure to run it from the root directory of the repository. Optionally, the model hyper-parameters can be configured by modifying the `configs\training\nonper-model_config.json` file. Similarly the dataset parameters can be configured by modifying the `configs\training\dataset_config.json` file. If not modified, the training will run on the default parameters which have been optimized by this repository's developers.

### Personalized training
Trains a non-personalized baseline model first, then fine-tunes that model for each user to generate multiple personalized models that are stored and retrieved later for recommendation. To train a personalized model, run the following command:
```bash
python trainers/train_personalized.py
```
This command will create and store a personalized model for each user in the database. Make sure to run it from the root directory of the repository. Optionally, the model hyper-parameters can be configured by modifying the `configs\training\per-model_config.json` file. Similarly the dataset parameters can be configured by modifying the `configs\training\dataset_config.json` file. If not modified, the training will run on the default parameters which have been optimized by this repository's developers.

## Data
The data for this project was obtained from the Synapse databse, [My Heart Counts](https://www.synapse.org/#!Synapse:syn16782070/tables/). Access to the database can be obtained via direct request to Synapse. Nonetheless, we provide some sample mock data in the `data/` directory. Our mock data is fully pre-processed, and we provide the option to select from 3 distinct augmentation methods: 
1) k-rolling average: replaces missing values with the rolling average of a k sized window along the temporal dimension for each user
2) KNN intra user imputation: searches for nearest neighbors only within the same user
3) KNN inter user imputation: searches for nearest neighbors accross all users

To select one of these optional augmentations, configure the`aug` parameter in the `configs\training\dataset_config.json` file before training.

To add more data to the system, simply repalce any of the csv files in the `data/` directory with an updated dataset. The only requisite for the system to work with the dataset is that the dataset follows the structure of user entries being rows, and features such as 'date', 'sleep_minutes', 'systolic_bp', etc being columns. As long the key column names (e.g., ['id', 'date']) and the target columns, that is the column names of the variable to predict (e.g., ['systolic', 'diastolic']), are specified as parameters in the `configs\training\dataset_config.json` and `configs\recommendation\query_config.json` files, the system will work with any sort of naming conventions.
### Raw Synapse data
Lastly, if interested in re-creating the dataset from the raw data in the Synapse database, we provide notebooks to download, pre-process and/or augment the data from the raw Synapse format. These notebooks can be found in the `synapse_data_processing` directory
