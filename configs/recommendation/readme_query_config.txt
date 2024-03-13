n = 5                               # Number of most important features to display
var_adjust = False                  # Whether to adjust recs according to total variance explained or not
verbose = True                      # Whether to print the recommendations and list of training entries for the user
key = ['healthCode', 'date']        # Columns to use as key
target = ['systolic', 'diastolic']  # Columns to predict