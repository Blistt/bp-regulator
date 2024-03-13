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
To get non-personalized recommendations, go to the "nonper_user_config.json" file in the "configs/recommendation" folder to enter the predictor variable values (e.g., sleep minutes, steps, historical_bp). Do the same with the "query_config.json" file located in the same folder for query parameters (e.g., how many features to recommend for).

Run the following command:
```bash
python _nonper_recommender.py
```

### Personalized recommendation
Personalized recommendations will use a distinct, fine-tuned model for each user. Meant for users already in the database, with a history of biometric and BP data on which a personalized model was fine tuned. To get personalized recommendations, go to the "per_user_config.json" file in the "configs/recommendation" folder to enter the ID of the user the recommendations are for. The system will retrieve the most recent predictor variable values from the database. Do the same with the "query_config.json" file located in the same folder for query parameters (e.g., how many features to recommend for).

Run the following command:
```bash
python _per_recommender.py
```

## Training


