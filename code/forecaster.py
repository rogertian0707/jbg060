# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:39 2019

@author: Hermii
"""
#%%

## SHOULD ADD ONlY HOLIDAY DAYS AS ONE-HOT ENCODED FEATURES IN DAY OF YEAR FEATURE
import pandas as pd
import glob
import holidays
#import datetime
from sklearn.ensemble import RandomForestRegressor
from numpy import mean
from data_bag import streets_rain, binary_rain, hourly_conversion, bound_dates
import random
from math import sqrt, ceil
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np

# PATHS

#Den Bosch flow
path = "../data/sewer_data/data_pump/RG8150/RG8150/"
path1 = "../data/sewer_data_db/data_wwtp_flow/RG1876_flow/"
path2 = "../data/sewer_data_db/data_wwtp_flow/RG1882_flow/"


#Bokhoven level
path3 = "../data/sewer_data/data_pump/RG8180_L0/"
#Bokhoven flow
path4 = "../data/sewer_data/data_pump/RG8180_Q0/"


#Haarsteeg level
path5 = "../data/sewer_data/data_pump/rg8170_N99/"
#Haarsteeg flow
path6 = "../data/sewer_data/data_pump/rg8170_99/"


#Helftheuvelweg level column 003 Helftheuvelweg *.csv
path7 = "../data/sewer_data_db/data_pump_level/"
#Helftheuvelweg flow 
path8 = "../data/sewer_data_db/data_pump_flow/1210FIT301_99/"


#Engelerschans level column “004 Engelerschans” *.csv
path9 = "../data/sewer_data_db/data_pump_level/"
#Engelerschans flow + Haarsteeg + Bokhoven, therefore substract for only Engeleschans
path10 = "../data/sewer_data_db/data_pump_flow/1210FIT201_99/"


#Maaspoort level Column: “006 Maaspoort” *.csv 
path11 = "../data/sewer_data_db/data_pump_level/"
#Maasport flow + Rompert
path12= "../data/sewer_data_db/data_pump_flow/1210FIT501_99/"


#Oude Engelenseweg level Column: “002 Oude Engelenseweg” *.csv
path13 = "../data/sewer_data_db/data_pump_level/"
#Oude Engelenseweg flow
path14 = "../data/sewer_data_db/data_pump_flow/1210FIT401_94/"


#De Rompert level Column: “005 de Rompert” *.csv
path15 = "../data/sewer_data_db/data_pump_level/"
#De Rompert flow + Maasport
path16 = "../data/sewer_data_db/data_pump_flow/1210FIT501_99/"

#Location linkage
path_linkinfo = "../data/sewer_model"
path_rain = "../data/sewer_data/rain_timeseries"

#Order of stations names is recurrent in the output dataframes here
station_names = ["Haarsteeg", "Bokhoven", "Hertogenbosch (Helftheuvelweg)",
                 "Hertogenbosch (Rompert)", "Hertogenbosch (Oude Engelenseweg)",
                 "Hertogenbosch (Maasport)"]

# Rain in each station df
rain_df = streets_rain(station_names, path_linkinfo, path_rain)

# List of dfs with each station's hourly rain classification in order of station_names list,
# with n = 15. (n=15 means rain_-15_class is "0" if no rain prior 15 hours and "1" otherwise)
hourly_rain_classified = binary_rain(station_names, rain_df, n=15)

# Level and flow of Bokhoven per hour (shapes match here, have to check if they do on other files,
# otherwise bound dates like in line 86) 

#Changed to Bokhoven, since there is another pump behind Haarsteeg at which if there is rain,
# it will influence the flow by a lot
level_bokhoven = hourly_conversion(path3, mean = True, need_concat=True)
flow_bokhoven = hourly_conversion(path4, mean = False, need_concat=True)

#level_haarsteg = hourly_conversion(path5, mean = True, need_concat=True)
#flow_haarsteeg = hourly_conversion(path6, mean = False, need_concat=True)

# Returns merged dataframe with the timestamps present in both dfs
flow_bokhoven  = bound_dates(flow_bokhoven, hourly_rain_classified[1], "datumBeginMeting", "Begin")

nl_holidays = holidays.CountryHoliday('NL')

def binary_holidays(country_holidays, dates):
    holiday = []
    for i in dates:
        if i in country_holidays:
            holiday.append(1)
        else:
            holiday.append(0)
    return holiday
    


def feature_setup(df_flow, df_level, country_holidays, name):
    
    dates = df_flow["Begin"]
    flow = df_flow["hstWaarde"]
    rained_n_class = df_flow["rain_-15_class"]
    rain = df_flow[name]
    #cumsum_previous_n = df_flow["cumsum_previous_15"]
    level = df_level["hstWaarde"]
    holidays_1 = binary_holidays(country_holidays, dates)
    
    features = pd.DataFrame()
    
    features["day_ofthe_month"] = dates.dt.day.astype(str)
    features["hour"] = dates.dt.hour.astype(str)
    features["day_ofthe_year"] = dates.dt.dayofyear.astype(str)
    features["day_ofthe_week"] = dates.dt.dayofweek.astype(str)
    features["holiday"] = holidays_1
    features["flow"] = flow
    
    # Add feature of amount of rain some timestamps before or during this hour (5 minute stamps)
    features["rain_hour"] = rain
    
    # Add a feature of level at previous timestamps before or during this hour (5 minute stamps)
    features["level"] = level
    
    # Not actual features, but columns by which we will filter prediction procedures
    #(Be sure to remove them before fitting)
    features["rain_N_ago"] = rained_n_class
    features["dates"] = dates
    
    # One-Hot encoding the categorical variables
    features = pd.get_dummies(features, columns = ["day_ofthe_month",
                                                    "hour",
                                                    "day_ofthe_year",
                                                    "day_ofthe_week",
                                                    "holiday"])


    return features

## check which ones are missing from .dropna()
df_features_bok = feature_setup(flow_bokhoven, level_bokhoven, nl_holidays, "Bokhoven").dropna()
#df_features_haar = feature_setup(flow_haarsteeg, level_haarsteg, nl_holidays, "Haarsteeg").dropna()

df_features_bok.to_csv("../data/Bokhoven_model_data.csv")
#df_features_haar.to_csv("../data/Haarsteeg_model_data.csv")


def mse(d):
    """Mean Squared Error"""
    return mean(d * d) 

def evaluate(model, predictions, test_labels):
    #predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print("RMSE = {}".format(sqrt(mse(errors))))
    
    return accuracy

def param_tuning_RS(features):
    """
    Random Search for parameter tuning (Needs generalization to other algorithms)
    """
    #Random search into Grid Search
    
    #RANDOM SEARCH
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    training_X, training_Y, testing_X, testing_Y = test_train(features)
    
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                                   n_iter = 100, cv = 3, verbose=2, random_state=42,
                                   n_jobs = -1)
    
    rf_random.fit(training_X, training_Y)
    
    print(rf_random.best_params_)
    
    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(training_X, training_Y)
    base_accuracy = evaluate(base_model, testing_X, testing_Y)
    
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, testing_X, testing_Y)
    
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))
    
    return  training_X, training_Y, testing_X, testing_Y 
    

def param_tuning_GS(features, base_accuracy):
    
    training_X, training_Y, testing_X, testing_Y  = features
    
    """
    Grid Search that should be manually tuned (the param_grid manually changed
    according to Random Search
    """
    #GRID SEARCH
    
    #Based on RS
    #Add external function to spit features
    # and add the base accuracy
    param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    
    grid_search.fit(train_features, train_labels)
    print(grid_search.best_params_)
    
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, test_features, test_labels)
    
    print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))
   

def random_forest(features, folds):
    """
    Random Forest algorithm (the model below could easily be replaced)
    with k-fold manually implemented stratified CV in order to balance out
    and have proportional number of dry and wet days in each partition
    """
    
    
    ### CV
    dry_indexes = list(features[features['rain_N_ago'] == 0].index)
    wet_indexes = list(features[features['rain_N_ago'] == 1].index)
    
    dry_n = len(dry_indexes)
    wet_n = len(wet_indexes)
    
    #Instruments from which 
    instr_dry = dry_indexes.copy()
    instr_wet = wet_indexes.copy()
    
    train_folds = []
    test_folds = []
    
    for i in range(0, folds):
        
        #if statements for the lengths at the last folds
        if len(instr_dry) < (dry_n/folds) or len(instr_wet) < (wet_n/folds):
            fold_i_test_dry = instr_dry
            fold_i_test_wet = instr_wet
            
            fold_i_test_dry.extend(fold_i_test_wet)
            test_folds.append(fold_i_test_dry)
            
            fold_i_train_dry = list(set(dry_indexes) - set(fold_i_test_dry))
            fold_i_train_wet = list(set(wet_indexes) - set(fold_i_test_wet))
            
            fold_i_train_dry.extend(fold_i_train_wet)
            train_folds.append(fold_i_train_dry)
            
        else:
            fold_i_test_dry = random.sample(instr_dry, ceil(dry_n/folds))
            instr_dry = list(set(instr_dry) - set(fold_i_test_dry))
            
            fold_i_test_wet = random.sample(instr_wet, ceil(wet_n/folds))
            instr_wet = list(set(instr_wet) - set(fold_i_test_wet))
            
            fold_i_test_dry.extend(fold_i_test_wet)
            test_folds.append(fold_i_test_dry)
            
            fold_i_train_dry = list(set(dry_indexes) - set(fold_i_test_dry))
            fold_i_train_wet = list(set(wet_indexes) - set(fold_i_test_wet))
            
            fold_i_train_dry.extend(fold_i_train_wet)
            train_folds.append(fold_i_train_dry)
    
    # Storing all MSEs
    all_folds_mse = []
    
    # Initializing empty column to lay the predictions back after the k-fold
    features_copy = features.copy()
    features_copy["predictions"] = 0
            
    for i in range(0, folds):
        df_train = features.loc[train_folds[i],:]
        df_test = features.loc[test_folds[i],:]
            
        train_Y = np.array(df_train["flow"])
        train_X = np.array(df_train.drop(["flow", "dates"], axis=1))
            
        test_Y = np.array(df_test["flow"])
        test_X = np.array(df_test.drop(["flow", "dates"], axis =1))
            
        # Here add the best parameters from RandomSearch and GridSearch
        rf = RandomForestRegressor(n_estimators = 400, min_samples_split=5,
                                       min_samples_leaf = 1,
                                       max_features = "sqrt", max_depth = 100, bootstrap = True)
        rf.fit(train_X, train_Y)
        #NEED to add a more elaborate evaluate function here
        test_predictions = rf.predict(test_X)
        error = test_Y - test_predictions
        mse_k = mse(error)
        rmse = sqrt(mse_k)
        all_folds_mse.append(mse_k)
        
        evaluate(rf, test_predictions, test_Y)
        
        #Store feature importances to see the most important features with rf.feature_importances_
        
        # Adding predictions back to the features
        features_copy.loc[test_folds[i], "predictions"] = test_predictions
        
    #Feature importance for one fold
    for i, k in zip(df_train.drop(["flow", "dates"], axis = 1).columns, rf.feature_importances_):
        print("Feature: {} ; Importance: {}".format(i, k))
        
    return features_copy
        
    
#df = random_forest(df_features, 10)


    
# =============================================================================

features["day"] = features["datumBeginMeting"].dt.day
features["hour"] = features["datumBeginMeting"].dt.hour
features["dayofyear"] = features["datumBeginMeting"].dt.dayofyear
features["dayofweek"] =  features["datumBeginMeting"].dt.dayofweek
# 
# 
# nl_holidays = holidays.CountryHoliday('NL')
# 
# holiday = []
# for i in features["datumBeginMeting"]:
#     if i in nl_holidays:
#         holiday.append(1)
#     else:
#         holiday.append(0)
# 
# features["holiday"] = holiday
# 
# features["level"] = level_haarsteeg["hstWaarde"]
#             
# train1 = features[(features["datumBeginMeting"] >= min(features["datumBeginMeting"])) &
#                  (features["datumBeginMeting"] <= pd.Timestamp(2019, 1, 1))]
# train = train1[["hour", "day","dayofweek", "level"]]
# test1 = features[features["datumBeginMeting"] > pd.Timestamp(2019, 1, 1)]
# test = test1[["hour", "day","dayofweek", "level"]]
# 
# 
# =============================================================================


# =============================================================================
# def test_train(features, dry = bool):
#     """
#     Spits out test and training data;
#     """
#     if dry == True:
#         features = features[df_features['rain_N_ago'] == 0]
#     else: 
#         features = features[df_features['rain_N_ago'] == 1]
#     
#     all_days = set(features["dates"].dt.date)
#     test_days = random.sample(all_days, 60)
#     training_days = [x for x in all_days if x not in test_days]
#     
#     training = features[features["dates"].dt.date.isin(training_days)]
#     training_X = training.drop(columns = ["flow", "dates"])
#     training_Y = training["flow"]
#     
#     testing = features[features["dates"].dt.date.isin(test_days)]
#     testing_X = testing.drop(columns = ["flow", "dates"])
#     testing_Y = testing["flow"]
#     
#     return training_X, training_Y, testing_X, testing_Y
# =============================================================================
