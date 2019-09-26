# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:48:09 2019

@author: Hermii
"""

    
import pandas as pd
import glob
pd.options.mode.chained_assignment = None 
#import datetime
#import seaborn as sns
#% matplotlib inline


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

#Missing Engelerschans (in map also)
station_names = ["Haarsteeg", "Bokhoven", "Hertogenbosch (Helftheuvelweg)",
                 "Hertogenbosch (Rompert)", "Hertogenbosch (Oude Engelenseweg)",
                 "Hertogenbosch (Maasport)"]

flow_paths = [path1, path2, path4, path6, path8, path10, path12, path14, path16]
level_paths = [path3, path5, path7, path9, path11, path13, path15]

def path_bag(flow, level, station_names):
    #station_to_flow = dict()
    #station_to_level = dict()
    return None


def streets_rain(station_names, path_linkinfo, path_rain):
    
    link = pd.read_csv(path_linkinfo+
                   "/20180717_dump riodat rioleringsdeelgebieden_matched_to_rainfall_locations.csv",
                   header = 9)
    
    rain = pd.concat([pd.read_csv(file, header = 2) for file in glob.glob(path_rain+"/*.*")], ignore_index = True)
    
    #Street names by stations
    streets = [list(link[(link["Naam kern"] == name)]["Naam / lokatie"]) for name in station_names]

    #Streets that are not found in rainfall data belonging to Hertogenbosch (Oude Engelenseweg)
    excl = ['Pettelaarpark', 'Geb. 16 Paleiskwartier']

    for s in streets:
        try:
            for i in excl:
                s.remove(i)
        except ValueError:
            pass


    #All the rain for the streets for the pump stations in order of station_names and the streets per station
    #can be found in streets nested list in the same order
    station_to_rain = rain[["Begin", "Eind"] + [i for sl in streets for i in sl]]

    station_rain = pd.DataFrame()
    for i in range(len(station_names)):
        station_rain[station_names[i]] = rain[streets[i]].sum(1)

    station_rain["Begin"] =  station_to_rain["Begin"]
    station_rain["End"] = station_to_rain["Eind"]
    station_rain["Begin"] = pd.to_datetime(station_rain["Begin"])
    station_rain["End"] = pd.to_datetime(station_rain["End"])
    station_rain["date"] = station_rain["Begin"].dt.date
    return station_rain.sort_values(by=["Begin"])

def labeler3(dayrain, previousdayrain):
    """
    Classifies a whole day as dry if previous OR the same day has less than 0.05
    """
    if dayrain <= 0.05 or previousdayrain <= 0.05:
        return 0
    else:
        return 1
        
def rainyday(df, station_names):
    """
    Returns a list with DFs per station for dry day classification
    """
    df_rainyday = df.groupby(df["Begin"].dt.date).sum().reset_index(drop=False)
    
    events_df = []
    
    for i in station_names:
        df_relevant = df_rainyday[[i, "Begin"]]
        df_relevant["previousday"] = df_relevant[i].shift(-1)
        df_relevant["dryday"] = df_relevant.apply(lambda x: labeler3(x[i], x["previousday"]), axis=1)
        events_df.append(df_relevant)
        
    return events_df
    

def labeler2(rain):
    """
    Classifies an hour as dry if there hasn't been rain previous n days days
    """
    if rain == 0:
        return "0"
    else:
        return "1"
    
def binary_rain(station_names, df, n):
    
    binary_rain_df = []
    df = df.sort_values(by="Begin")
    df = df.set_index("Begin", drop = False).resample("60Min").sum()
    df = df.reset_index(drop=False)
    
    for i in station_names:
        df_relevant = df[[i, "Begin"]]
        df_relevant["cumsum_previous_15"] = df[i].rolling(min_periods=1, window=n).sum()
        df_relevant["rain_-15_class"] = df_relevant.apply(lambda x: labeler2(x["cumsum_previous_15"]), axis=1)
        binary_rain_df.append(df_relevant)
    return binary_rain_df

# Here n = 15
#events_df = rain_events2(station_names, df, 15)


counter = 0
def labeler(rain, previous_rain):
    """
    Labels rain events (events are defined as consecutive measurements where there is no stop of rain)
    (Previous definition)
    """
    global counter
    if rain == 0:
        return "None"
    else:
        if previous_rain == 0:
            counter += 1
            return counter
        if previous_rain != 0:
            return counter
        
def rain_events(station_names, df):
    events_df = []
    df = df.sort_values(by="Begin")
    
    for i in station_names:
        df_relevant = df[[i, "Begin", "End"]]
        df_relevant["instr"] = df_relevant[i].shift()
        df_relevant["rain_event_ID"] = df_relevant.apply(lambda x: labeler(x[i], x["instr"]), axis=1)
        df_relevant.drop(columns=["instr"], inplace=True)
        events_df.append(df_relevant)
    return events_df

#events_df = rain_events(station_names, df)

def group_by_event(events_df, station_name, rain_event_col, k):
    df = events_df[k]
    
    #Filtering rain events where there was no rain
    df = df[df[rain_event_col] != "None"]
    
    #should be elsewhere
    df["End"] = pd.to_datetime(df["End"])
    #
    
    #Converting duration of individual measurements to minutes
    df["duration"] = (df["End"] - df["Begin"]).dt.total_seconds()/60
    
    #Filter measurements that last more than 1 week (there are errors in data)
    df = df[(df["duration"] <= 10080) & (df["duration"] >= 0)]
    
    #Group by rain event ID and sum the durations and rain measurements
    df = df[["duration", station_name, rain_event_col]].groupby(rain_event_col).sum()
    return df


def bound_dates(df1, df2, df1_datecol, df2_datecol):
    
    # Making sure both have the same range of dates
    df1[df1_datecol] = pd.to_datetime(df1[df1_datecol])
    df2[df2_datecol] = pd.to_datetime(df2[df2_datecol])
    
    df1 = pd.merge(left=df1, left_on=df1_datecol,
         right=df2, right_on=df2_datecol)
    #d2 = pd.merge(left=df2, left_on=df2_datecol,
    #     right=df1, right_on=df1_datecol)
    df1.drop(columns=[df2_datecol])
    return df1

def hourly_conversion(path):
    """
    Converts data to hourly format
    """
    df = pd.concat([pd.read_csv(file) for file in glob.glob(path+"/*.*")], ignore_index = True)
    df = df[["datumBeginMeting", "datumEindeMeting", "hstWaarde"]].sort_values(by='datumBeginMeting')
    df["datumBeginMeting"] = pd.to_datetime(df["datumBeginMeting"])
    df["datumEindeMeting"] = pd.to_datetime(df["datumEindeMeting"])
    df = df.set_index("datumBeginMeting", drop = False).resample("60Min").sum()
    df = df.reset_index()
    return df
    