# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:53:35 2019

@author: Hermii
"""

from data_bag import hourly_conversion, streets_rain, binary_rain, bound_dates
from forecaster import random_forest, feature_setup
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import math

path_viz = "../graphs/all_script"

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

#Missing Engelerschans (in map also)



print("Running saving flow files")
### ASK IF IT IS THOUSANDS OR SOME KIND OF ERROR

# Comment this out if you haven't run it yet

# =============================================================================
# df = pd.concat([pd.read_csv(file, delimiter = ";", dtype={'001: Poeldonk Neerslagmeting (mm)': str}) for file in glob.glob(path7+"/*.csv")], ignore_index = True)
# #df['hstWaarde'] = df['hstWaarde'].str.replace(',', '').astype(float)
# print("Concatinated")
# # # #LEVELS
# df["datumBeginMeting"] = df["Datum"] + " " + df["Tijd"]
# df["datumBeginMeting"] = pd.to_datetime(df["datumBeginMeting"])
# print("Date columnb saved")
# #   
# df_Helftheuvelweg_level = df[["datumBeginMeting", "003: Helftheuvelweg Niveau (cm)"]].rename(columns={"003: Helftheuvelweg Niveau (cm)": "hstWaarde"})
# df_Helftheuvelweg_level["hstWaarde"] = df_Helftheuvelweg_level['hstWaarde'].str.replace(',', '').astype(float)
# df_Helftheuvelweg_level.to_csv(path7 + "Hertogenbosch (Helftheuvelweg).csv")
# 
# 
# #  
# df_De_Rompert_level = df[["datumBeginMeting", '005: De Rompert Niveau (cm)']].rename(columns = {'005: De Rompert Niveau (cm)':"hstWaarde"})
# df_De_Rompert_level["hstWaarde"] = df_De_Rompert_level["hstWaarde"].str.replace(',', '').astype(float)
# df_De_Rompert_level.to_csv(path9 + "Hertogenbosch (Rompert).csv")
# 
# #  
# df_Oude_Engelenseweg_level = df[["datumBeginMeting", '002: Oude Engelenseweg Niveau actueel (1&2)(cm)']].rename(columns = {'002: Oude Engelenseweg Niveau actueel (1&2)(cm)':"hstWaarde"})
# df_Oude_Engelenseweg_level["hstWaarde"] = df_Oude_Engelenseweg_level['hstWaarde'].str.replace(',', '').astype(float)
# df_Oude_Engelenseweg_level.to_csv(path13 + "Hertogenbosch (Oude Engelenseweg).csv")
# 
# #   
# df_Maasport_level = df[["datumBeginMeting", '006: Maaspoort Niveau actueel (1&2)(cm)']].rename(columns = {'006: Maaspoort Niveau actueel (1&2)(cm)':"hstWaarde"})
# df_Maasport_level["hstWaarde"] = df_Maasport_level['hstWaarde'].str.replace(',', '').astype(float)
# df_Maasport_level.to_csv(path13 + "Hertogenbosch (Maasport).csv")
# =============================================================================


# =============================================================================
# =============================================================================


# =============================================================================
# We would also consider the risks of a Leave-One-Out Cross Validation (as well as predicting
#    every data point with the whole rest of either dry data with only features for 
#    hour of the day, day of the week and day of the month or wet data with the course 
#    of month and rain measurements more, which are probably not that high since we do not 
#    have the data over more than two years.
# =============================================================================


print("Done writing files for flow")

station_names = ["Haarsteeg", "Bokhoven", "Hertogenbosch (Helftheuvelweg)",
                 "Hertogenbosch (Rompert)", "Hertogenbosch (Oude Engelenseweg)",
                 "Hertogenbosch (Maasport)"]

nl_holidays = holidays.CountryHoliday('NL')

def AddColumnsPred(df):
    '''
    Input:      - df with predictions per pump
    Output:     - df with extra features
    '''
    df['datetime'] = pd.to_datetime(df['dates'])
    df['Date'] = pd.DatetimeIndex(df.datetime).normalize()
    df['Weekday'] = df['datetime'].dt.day_name()
    df['Hour'] = df['datetime'].dt.hour
    df['TimeOfDay'] = df['datetime'].dt.time
    df['Year'] = df['datetime'].dt.year
    df['Month'] = df['datetime'].dt.month_name()
    df['Day'] = df['datetime'].dt.day
    df['DayofYear'] = df['datetime'].dt.dayofyear
    
    return df


def PrepareDfPred(df):
    '''
    Input:      - df with predictions per pump
    Output:     - df with only the necesary columns
    '''
    # remove all features
    df = df[['flow','rain_hour', 'level', 'rain_N_ago', 'dates', 'predictions']]
    
    # add columns with extra features
    df = AddColumnsPred(df)
   
    # add error columns
    df['error'] = df['flow'] - df['predictions']
    df['sq_error'] = df['error']**2
    df['rse'] = abs(df['error'])
    return df

    
def DryPred(df):
    '''
    Input:      - df with predictions per pump
                - string of the pump name
    Output:     - df with only dry hours according to our definition
    '''
    dry = df[df['rain_N_ago'] == 0]
    return dry

def RainyPred(df):
    '''
    Input:      - df with predictions per pump
                - string of the pump name
    Output:     - df with only rainy hours according to our definition
    '''
    wet = df[df['rain_N_ago'] == 1]
    return wet

def SeasonalDf(df):
    '''
    Input:      - df with data per pump
    Output:     - list of 4 df's so a df per season
    '''
    winter = df.loc[(df["Month"] == 'December') | (df["Month"] == 'January') | (df["Month"] == 'February')]
    spring = df.loc[(df["Month"] == 'March') | (df["Month"] == 'April') | (df["Month"] == 'May')]
    summer = df.loc[(df["Month"] == 'June') | (df["Month"] == 'July') | (df["Month"] == 'August')]
    autumn = df.loc[(df["Month"] == 'September') | (df["Month"] == 'October') | (df["Month"] == 'November')]
    
    season_dfs = [winter, spring, summer, autumn]
    return season_dfs


################################################################################################
# A FUNCTION TO DO ALL THE ERROR VISUALIZATIONS

def DoAllErrorVis(df, pump_name ):
    '''
    Input:      - file together with the path: '../data/file name here'
                    - (for now i use the code from forecaster.py to generate csv's for pumps and save them in the data folder
                - string of the pump name (to access the rain column)
    Output:     - a bunch of plots hopefully
    '''
    # read in the prediction data
    pred_df = df
    
    # prepare the df for visualizations
    pred_df = PrepareDfPred(pred_df)
    
    # split between dry and rainy
    dry = DryPred(pred_df)
    rainy = RainyPred(pred_df)
    
    # do all visualizations for dry time predictions
    RMSEperHour(dry, pump_name, 'all')
    RMSEperMonth(dry, pump_name, 'all')
    RMSEperDay(dry, pump_name, 'all')
    RMSEoverYear(dry, pump_name, 'all')
    SeasonalError(dry, pump_name, 'all')
    ErrorHist(dry, pump_name, "all")
    
    RMSEperHour(rainy, pump_name, 'all')
    RMSEperMonth(rainy, pump_name, 'all')
    RMSEperDay(rainy, pump_name, "all")
    RMSEoverYear(rainy, pump_name, 'all')
    SeasonalError(rainy, pump_name, 'all')
    ErrorHist(rainy, pump_name, "all")
    
def RMSEperHour(df, pump_name, condition):
    
    # overall mse
    rmse = np.sqrt(df['sq_error'].sum() / len(df['sq_error']))
    
    # group data by hour to obtain the MSE per each hour (as add all squared errors and then take the mean)
    hour_group = df.groupby('Hour')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.plot(hour_group.index, np.sqrt(hour_group['sq_error']), label='RMSE per hour');
    ax.axhline(y=rmse, linewidth = 2, color='red', label = 'RMSE over all ' + condition + ' day predictions: {:.1f}'.format(rmse))
    ax.xaxis.set_ticks(np.arange(min(hour_group.index), max(hour_group.index)+1, 1.0))
    ax.set_ylabel('Root mean squared error', fontsize=15)
    ax.set_xlabel('Hour of the day', fontsize=15)
    ax.set_title(pump_name + ': Change in RMSE per hour on ' + condition + ' days', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.legend(prop={'size': 14});
    fig.savefig(path_viz + pump_name + condition + " per_hour.png")
    
    
def RMSEperMonth(df, pump_name, condition):
    
    # overall rmse
    rmse = np.sqrt(df['sq_error'].sum() / len(df['sq_error']))
    
    # group data by month to obtain the MSE per each month
    g = df.groupby('Month', sort=False)[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.bar(g.index, np.sqrt(g['sq_error']));
    ax.axhline(y=rmse, linewidth = 2, color='red', label = 'RMSE over all ' + condition + ' day predictions: {:.1f}'.format(rmse))
    ax.set_ylabel('Root mean squared error', fontsize=15)
    ax.set_xlabel('Month', fontsize=15)
    ax.set_title(pump_name + ': RMSE of predictions per month on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    plt.xticks(rotation=45);
    fig.savefig(path_viz + pump_name + condition + " per_month.png")
    

def RMSEperDay(df, pump_name, condition):
    
    # overall mse
    rmse = np.sqrt(df['sq_error'].sum() / len(df['sq_error']))
    
    # group data by day of the month to obtain the MSE per each day in a month
    g = df.groupby('Day')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.plot(g.index, np.sqrt(g['sq_error']), label='RMSE per day');
    ax.axhline(y=rmse, linewidth = 2, color='red', label = 'RMSE over all ' + condition + ' day predictions: {:.1f}'.format(rmse))
    ax.set_ylabel('Root mean squared error', fontsize=15)
    ax.set_xlabel('Day of the month', fontsize=15)
    ax.set_title(pump_name + ': RMSE of predictions per day over all months on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.xaxis.set_ticks(np.arange(min(g.index), max(g.index)+1, 2.0))
    fig.savefig(path_viz + pump_name + condition + " per_day.png")
    

def RMSEoverYear(df, pump_name, condition):
    
    # overall mse
    rmse = np.sqrt(df['sq_error'].sum() / len(df['sq_error']))
    
    # group data by day of the year to obtain the MSE per each day in the year
    g = df.groupby('DayofYear')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.plot(g.index, np.sqrt(g['sq_error']), label='RMSE per day');
    ax.axhline(y=rmse, linewidth = 2, color='red', label = 'MSE over all ' + condition + ' day predictions: {:.1f}'.format(rmse))
    ax.set_ylabel('Root mean squared error', fontsize=15)
    ax.set_xlabel('Day of the year', fontsize=15)
    ax.set_title(pump_name + ': RMSE of predictions per day over a year on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.xaxis.set_ticks(np.arange(min(g.index), max(g.index)+1, 50.0));
    
    # adding some colour to the plot background to distinguish between months
    ax.axvspan(1, 31, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(31, 59, facecolor='pink', alpha=0.2)
    ax.axvspan(59, 90, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(90, 121, facecolor='pink', alpha=0.2)
    ax.axvspan(121, 151, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(151, 182, facecolor='pink', alpha=0.2)
    ax.axvspan(182, 213, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(213, 244, facecolor='pink', alpha=0.2)
    ax.axvspan(244, 274, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(274, 305, facecolor='pink', alpha=0.2)
    ax.axvspan(305, 335, facecolor='skyblue', alpha=0.2)
#     ax.axvspan(335, 365, facecolor='pink', alpha=0.2);

    anno_place = [1,31,59,90,121,151,182,213,244,274,305,335]
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December']
    for i in range(12):
        ax.annotate(months[i], (anno_place[i], 350), xytext=(10, 10), textcoords='offset points');
    ax.margins(0);
    fig.savefig(path_viz + pump_name + condition + " over_year.png")


def SeasonalError(df, pump_name, condition):

    # group data by season
    seasonal_pred = SeasonalDf(df)
    # season names for convenience
    season_names = ['winter', 'spring', 'summer', 'autumn']
    
    # colors to be used for each season on the plot
    colors = ['royalblue', 'green', 'darkorange', 'red']
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    ax.grid()

    for i in range(4):
        # group data by day of the month
        g = seasonal_pred[i].groupby('Day')[['rse']].mean()

        ax.plot(g.index, g['rse'], label = season_names[i], color = colors[i])

    # overall rmse
    rmse = np.sqrt(df['sq_error'].sum() / len(df['sq_error']))
    ax.axhline(y=rmse, linewidth = 3, color='black', label = 'RMSE over all dry day predictions: {:.1f}'.format(rmse))
    
    ax.set_ylabel("Root squared error", fontsize=15)
    ax.set_xlabel("Day of the month", fontsize=15)
    ax.legend(prop={'size': 14})
    ax.set_title(pump_name + ': Root squared error over the course of an average month in a season on ' + condition + ' days', fontsize=18);
    ax.xaxis.set_ticks(np.arange(min(g.index), max(g.index)+1, 2.0));
    fig.savefig(path_viz + pump_name + condition + " per_season.png")
    
def ErrorHist(df, pump_name, condition):
  
    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()

    sns.distplot(df['error'], bins=100, color = 'darkblue');
    
    ax.set_xlabel('Difference between actual and predicted values', fontsize=15)
    ax.set_title(pump_name + ': Distribution of prediction variation on ' + condition + ' days', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=14);
    fig.savefig(path_viz + pump_name + condition + " error_hist.png")



def run_forecast_vis(station_names, path5, path6, path3,
                     path4, path7, path8, path9, path16, path13,
                     path14, path12, path_linkinfo, path_rain, nl_holidays):
    levels = []
    flows = []
    
    RMSE = []
    
    level_haarsteeg = hourly_conversion(path5, mean = True, need_concat = True)
    flow_haarsteeg = hourly_conversion(path6, mean = False, need_concat = True)
    
    
    level_bokhoven = hourly_conversion(path3, mean = True, need_concat = True)
    flow_bokhoven = hourly_conversion(path4, mean = False, need_concat = True)
    print("Problem?")
    level_helftheuvelweg = hourly_conversion(path7 + "Hertogenbosch (Helftheuvelweg).csv", mean = True, need_concat = False)
    flow_helftheuvelweg = hourly_conversion(path8, mean = False, need_concat = True)
    print("No")
    level_derompert = hourly_conversion(path9 + "Hertogenbosch (Rompert).csv", mean = True)
    flow_derompert = hourly_conversion(path16, mean = False, need_concat = True)
    
    level_oudeengelenseweg = hourly_conversion(path13 + "Hertogenbosch (Oude Engelenseweg).csv", mean = True, need_concat = False)
    flow_oudeengelenseweg = hourly_conversion(path14, mean = False, need_concat = True)
    
    level_maasport = hourly_conversion(path13 + "Hertogenbosch (Maasport).csv", mean = True, need_concat = False)
    flow_maasport = hourly_conversion(path12, mean = False, need_concat = True)
    
    rain_df = streets_rain(station_names, path_linkinfo, path_rain)
    hourly_rain_classified = binary_rain(station_names, rain_df, n=15)
    
    levels.extend([level_haarsteeg, level_bokhoven, level_helftheuvelweg, level_derompert, level_oudeengelenseweg, level_maasport])
    flows.extend([flow_haarsteeg, flow_bokhoven, flow_helftheuvelweg, flow_derompert, flow_oudeengelenseweg, flow_maasport])
    
    print("done storing flows and levels in run_forecast_vis function")
    
    for k, name in enumerate(station_names):
        df = bound_dates(flows[k], hourly_rain_classified[k], "datumBeginMeting", "Begin")
        df_features = feature_setup(df, levels[k], nl_holidays, name).dropna()
        mse_s, df = random_forest(df_features, 10)
        RMSE.append(math.sqrt(mse_s))
        print("Starting viz")
        #DoAllErrorVis(df, name)
        print("Done with "+name)
        
    for i, d in enumerate(station_names):
        print(d + " RMSE: " + RMSE[i])
        
run_forecast_vis(station_names, path5, path6, path3, path4, path7, path8, path9,
                 path16, path13, path14, path12, path_linkinfo, path_rain, nl_holidays)
        
#level_bokhoven = hourly_conversion(path3, mean = True, need_concat = True)
#flow_bokhoven = hourly_conversion(path4, mean = False, need_concat = True)



# =============================================================================
# rain_df = streets_rain(station_names, path_linkinfo, path_rain)
# hourly_rain_classified = binary_rain(station_names, rain_df, n=15)
# df = bound_dates(flow_bokhoven, hourly_rain_classified[1], "datumBeginMeting", "Begin")
# df_features = feature_setup(df, level_bokhoven, nl_holidays, "Bokhoven").dropna()
# mse_s, df = random_forest(df_features, 10)
# # =============================================================================
# =============================================================================
#print("Starting viz")
#DoAllErrorVis(df, "Bokhoven")