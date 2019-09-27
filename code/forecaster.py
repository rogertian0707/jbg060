# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:08:39 2019

@author: Hermii
"""
#%%
import pandas as pd
import glob
import holidays
#import datetime
from sklearn.ensemble import RandomForestRegressor
from numpy import mean
from data_bag import streets_rain, binary_rain, hourly_conversion, bound_dates



# PATHS

#Den Bosch flow
path = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/data_pump/RG8150/RG8150/"
path1 = "../data/waterschap-aa-en-maas_sewage_2019_db_pumps/sewer_data_db/data_wwtp_flow/RG1876_flow/"
path2 = "../data/waterschap-aa-en-maas_sewage_2019_db_pumps/sewer_data_db/data_wwtp_flow/RG1882_flow/"


#Bokhoven level
path3 = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/data_pump/RG8180_L0/"
#Bokhoven flow
path4 = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/data_pump/RG8180_Q0/"


#Haarsteeg level
path5 = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/data_pump/rg8170_N99/"
#Haarsteeg flow
path6 = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/data_pump/rg8170_99/"


#Helftheuvelweg level column 003 Helftheuvelweg *.csv
path7 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_level/"
#Helftheuvelweg flow 
path8 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_flow/1210FIT301_99/"


#Engelerschans level column “004 Engelerschans” *.csv
path9 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_level/"
#Engelerschans flow + Haarsteeg + Bokhoven, therefore substract for only Engeleschans
path10 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_flow/1210FIT201_99/"


#Maaspoort level Column: “006 Maaspoort” *.csv 
path11 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_level/"
#Maasport flow + Rompert
path12= "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_flow/1210FIT501_99/"


#Oude Engelenseweg level Column: “002 Oude Engelenseweg” *.csv
path13 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_level/"
#Oude Engelenseweg flow
path14 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_flow/1210FIT401_94/"


#De Rompert level Column: “005 de Rompert” *.csv
path15 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_level/"
#De Rompert flow + Maasport
path16 = "../data/waterschap-aa-en-maas_sewage_2019_db/sewer_data_db/data_pump_flow/1210FIT501_99/"

#Location linkage
path_linkinfo = "../data/waterschap-aa-en-maas_sewage_2019/sewer_model"
path_rain = "../data/waterschap-aa-en-maas_sewage_2019/sewer_data/rain_timeseries"

station_names = ["Haarsteeg", "Bokhoven", "Hertogenbosch (Helftheuvelweg)",
                 "Hertogenbosch (Rompert)", "Hertogenbosch (Oude Engelenseweg)",
                 "Hertogenbosch (Maasport)"]

# Rain in each station df
rain_df = streets_rain(station_names, path_linkinfo, path_rain)

# List of dfs with each station's hourly rain classification in order of station_names list,
# with n = 15. (n=15 means rain_-15_class is "0" if no rain prior 15 hours and "1" otherwise)
hourly_rain_classified = binary_rain(station_names, rain_df, n=1)

# Level and flow of Haarsteeg per hour (shapes match here, have to check if they do on other files,
# otherwise bound dates like in line 86)
level_haarsteeg = hourly_conversion(path5, mean = True)
flow_haarsteeg = hourly_conversion(path6, mean = False)

# Returns merged dataframe with the timestamps present in both dfs
flow_haarsteeg  = bound_dates(flow_haarsteeg, hourly_rain_classified[0], "datumBeginMeting", "Begin")

nl_holidays = holidays.CountryHoliday('NL')

def binary_holidays(country_holidays, dates):
    holiday = []
    for i in dates:
        if i in nl_holidays:
            holiday.append(1)
        else:
            holiday.append(0)
    return holiday
    


def feature_setup(df_flow, df_level, country_holidays):
    
    dates = df_flow["Begin"]
    flow = df_flow["hstWaarde"]
    rained_n_class = df_flow["rain_-15_class"]
    rain = df_flow["Haarsteeg"]
    #cumsum_previous_n = df_flow["cumsum_previous_15"]
    level = df_level["hstWaarde"]
    holidays = binary_holidays(nl_holidays, dates)
    
    features = pd.DataFrame()
    
    features["day_ofthe_month"] = dates.dt.day
    features["hour"] = dates.dt.hour
    features["day_ofthe_year"] = dates.dt.dayofyear
    features["day_ofthe_week"] = dates.dt.dayofweek
    features["holiday"] = holidays
    features["flow"] = flow
    
    # Add feature of amount of rain some timestamps before or during this hour (5 minute stamps)
    features["rain_hour"] = rain
    
    # Add a feature of level at previous timestamps before or during this hour (5 minute stamps)
    features["level"] = level
    
    #Not actual features, but columns by which we will filter prediction procedures
    #(Be sure to remove them before fitting)
    features["rain_N_ago"] = rained_n_class
    
    return features

df_features = feature_setup(flow_haarsteeg, level_haarsteeg, nl_holidays)
    
    
    
    
    
# =============================================================================

# features["day"] = features["datumBeginMeting"].dt.day
# features["hour"] = features["datumBeginMeting"].dt.hour
# features["dayofyear"] = features["datumBeginMeting"].dt.dayofyear
# features["dayofweek"] =  features["datumBeginMeting"].dt.dayofweek
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
# rf = RandomForestRegressor(n_estimators = 1000)
# rf.fit(train, train1["hstWaarde"])
# prediction = rf.predict(test)
# error = test1["hstWaarde"] - prediction
# def mse(d):
#     """Mean Squared Error"""
#     return mean(d * d)
# 
# mse = mse(error)
# print(mse)
# =============================================================================
