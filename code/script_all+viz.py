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
#df = pd.concat([pd.read_csv(file, delimiter = ";", dtype={'001: Poeldonk Neerslagmeting (mm)': str}) for file in glob.glob(path7+"/*.csv")], ignore_index = True)
# #df['hstWaarde'] = df['hstWaarde'].str.replace(',', '').astype(float)
#print("Concatinated")
# # # #LEVELS
#df["datumBeginMeting"] = df["Datum"] + " " + df["Tijd"]
#df["datumBeginMeting"] = pd.to_datetime(df["datumBeginMeting"])
#print("Date columnb saved")
# #   
#df_Helftheuvelweg_level = df[["datumBeginMeting", "003: Helftheuvelweg Niveau (cm)"]].rename(columns={"003: Helftheuvelweg Niveau (cm)": "hstWaarde"})
#df_Helftheuvelweg_level["hstWaarde"] = df_Helftheuvelweg_level['hstWaarde'].str.replace(',', '').astype(float)
#df_Helftheuvelweg_level.to_csv(path7 + "Hertogenbosch (Helftheuvelweg).csv")
# 
# 
# #  
#df_De_Rompert_level = df[["datumBeginMeting", '005: De Rompert Niveau (cm)']].rename(columns = {'005: De Rompert Niveau (cm)':"hstWaarde"})
#df_De_Rompert_level["hstWaarde"] = df_De_Rompert_level["hstWaarde"].str.replace(',', '').astype(float)
#df_De_Rompert_level.to_csv(path9 + "Hertogenbosch (Rompert).csv")
# 
# #  
#df_Oude_Engelenseweg_level = df[["datumBeginMeting", '002: Oude Engelenseweg Niveau actueel (1&2)(cm)']].rename(columns = {'002: Oude Engelenseweg Niveau actueel (1&2)(cm)':"hstWaarde"})
#df_Oude_Engelenseweg_level["hstWaarde"] = df_Oude_Engelenseweg_level['hstWaarde'].str.replace(',', '').astype(float)
#df_Oude_Engelenseweg_level.to_csv(path13 + "Hertogenbosch (Oude Engelenseweg).csv")

 #   
#df_Maasport_level = df[["datumBeginMeting", '006: Maaspoort Niveau actueel (1&2)(cm)']].rename(columns = {'006: Maaspoort Niveau actueel (1&2)(cm)':"hstWaarde"})
#df_Maasport_level["hstWaarde"] = df_Maasport_level['hstWaarde'].str.replace(',', '').astype(float)
#df_Maasport_level.to_csv(path13 + "Hertogenbosch (Maasport).csv")
# =============================================================================


# =============================================================================
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
    MSEperHour(dry, pump_name, 'dry')
    MSEperMonth(dry, pump_name, 'dry')
    MSEperDay(dry, pump_name, 'dry')
    MSEoverYear(dry, pump_name, 'dry')
    SeasonalError(dry, pump_name, 'dry')
    
    MSEperHour(rainy, pump_name, 'wet')
    MSEperMonth(rainy, pump_name, 'wet')
    MSEperDay(rainy, pump_name, 'wet')
    MSEoverYear(rainy, pump_name, 'wet')
    SeasonalError(rainy, pump_name, 'wet')
    
def MSEperHour(df, pump_name, condition):
    
    # overall mse
    mse = df['sq_error'].sum() / len(df['sq_error'])
    
    # group data by hour to obtain the MSE per each hour (as add all squared errors and then take the mean)
    hour_group = df.groupby('Hour')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.plot(hour_group.index, hour_group['sq_error'], label='MSE per hour');
    ax.axhline(y=mse, linewidth = 2, color='red', label = 'MSE over all ' + condition + ' day predictions: {:.1f}'.format(mse))
    ax.xaxis.set_ticks(np.arange(min(hour_group.index), max(hour_group.index)+1, 1.0))
    ax.set_ylabel('Mean squared error', fontsize=15)
    ax.set_xlabel('Hour of the day', fontsize=15)
    ax.set_title(pump_name + ': Change in MSE per hour on ' + condition + ' days', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.legend(prop={'size': 14});
    fig.savefig(pump_name + condition + " per_hour.png")
    
    
def MSEperMonth(df, pump_name, condition):
    
    # overall mse
    mse = df['sq_error'].sum() / len(df['sq_error'])
    
    # group data by month to obtain the MSE per each month
    g = df.groupby('Month', sort=False)[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.bar(g.index, g['sq_error']);
    ax.axhline(y=mse, linewidth = 2, color='red', label = 'MSE over all ' + condition + ' day predictions: {:.1f}'.format(mse))
    ax.set_ylabel('Mean squared error', fontsize=15)
    ax.set_xlabel('Month', fontsize=15)
    ax.set_title(pump_name + ': MSE of predictions per month on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    plt.xticks(rotation=45);
    fig.savefig(pump_name + condition + " per_month.png")
    

def MSEperDay(df, pump_name, condition):
    
    # overall mse
    mse = df['sq_error'].sum() / len(df['sq_error'])
    
    # group data by day of the month to obtain the MSE per each day in a month
    g = df.groupby('Day')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.plot(g.index, g['sq_error'], label='MSE per day');
    ax.axhline(y=mse, linewidth = 2, color='red', label = 'MSE over all ' + condition + ' day predictions: {:.1f}'.format(mse))
    ax.set_ylabel('Mean squared error', fontsize=15)
    ax.set_xlabel('Day of the month', fontsize=15)
    ax.set_title(pump_name + ': MSE of predictions per day over all months on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.xaxis.set_ticks(np.arange(min(g.index), max(g.index)+1, 2.0))
    fig.savefig(pump_name + condition + " per_day.png")
    

def MSEoverYear(df, pump_name, condition):
    
    # overall mse
    mse = df['sq_error'].sum() / len(df['sq_error'])
    
    # group data by day of the year to obtain the MSE per each day in the year
    g = df.groupby('DayofYear')[['sq_error']].mean()

    fig, ax = plt.subplots(figsize=(15,10))
    ax.grid()
    ax.set_axisbelow(True)
    ax.plot(g.index, g['sq_error'], label='MSE per day');
    ax.axhline(y=mse, linewidth = 2, color='red', label = 'MSE over all ' + condition + ' day predictions: {:.1f}'.format(mse))
    ax.set_ylabel('Mean squared error', fontsize=15)
    ax.set_xlabel('Day of the year', fontsize=15)
    ax.set_title(pump_name + ': MSE of predictions per day over a year on ' + condition + ' days', fontsize=18);
    ax.legend(prop={'size': 14});
    ax.tick_params(axis='both', which='major', labelsize=14);
    ax.xaxis.set_ticks(np.arange(min(g.index), max(g.index)+1, 50.0));
    fig.savefig(pump_name + condition + " over_year.png")


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
    fig.savefig(pump_name + condition + " per_season.png")







def run_forecast_vis(station_names, path5, path6, path3,
                     path4, path7, path8, path9, path16, path13,
                     path14, path12, path_linkinfo, path_rain, nl_holidays):
    levels = []
    flows = []
    
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
    return flows
#    for k, name in enumerate(station_names):
#        df = bound_dates(flows[k], hourly_rain_classified[k], "datumBeginMeting", "Begin")
#        df_features = feature_setup(df, levels[k], nl_holidays, name).dropna()
#        df = random_forest(df_features, 10)
#        print("Starting viz")
#        DoAllErrorVis(df, name)
#        print("Done with "+name)
        
        
run_forecast_vis(station_names, path5, path6, path3, path4, path7, path8, path9,
                 path16, path13, path14, path12, path_linkinfo, path_rain, nl_holidays)
        
        
    