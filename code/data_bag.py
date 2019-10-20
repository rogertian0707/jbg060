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
<<<<<<< Updated upstream
    
=======
    """
    Function to link the station streets to the rain on those streets.
    """
>>>>>>> Stashed changes
    link = pd.read_excel(path_linkinfo+
                   "/20180717_dump riodat rioleringsdeelgebieden_matched_to_rainfall_locations.xlsx",
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
    threshold = 0.05
    if rain >= 0 and rain <= threshold:
        return 0
    else:
        return 1
    
def last_n_cumsum(n, name, df):
    B = []
    i =0
    n_lim = n
    station = list(df[name])
    while i<len(station):
        if i<n_lim:
            B.append(sum(station[0:i+1]))
        if i>=n_lim:
            B.append(sum(station[i-n_lim:i+1]))
        i=i+1
    return B

def binary_rain(station_names, df, n):
    
    binary_rain_df = []
    df = df.sort_values(by="Begin")
    df = df.set_index("Begin", drop = False).resample("60Min").sum()
    df = df.reset_index(drop=False)
    
    for i in station_names:
        df_relevant = df[[i, "Begin"]]
        df_relevant["cumsum_previous_15"] = last_n_cumsum(n, i, df)
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

def hourly_conversion(path, mean=bool, need_concat=bool):
    """
    Converts data to hourly format
    """
    if need_concat == True:
       df = pd.concat([pd.read_csv(file) for file in glob.glob(path+"/*.*")], ignore_index = True)
    else:
        df = pd.read_csv(path)
    
    df = df[["datumBeginMeting", "hstWaarde"]].sort_values(by='datumBeginMeting') #, "datumEindeMeting"
    df["datumBeginMeting"] = pd.to_datetime(df["datumBeginMeting"])
    #df["datumEindeMeting"] = pd.to_datetime(df["datumEindeMeting"])
    if mean == True:
        df = df.set_index("datumBeginMeting", drop = False).resample("60Min").mean()
    else:
        df = df.set_index("datumBeginMeting", drop = False).resample("60Min").sum()
   
    df = df.reset_index()
    return df

class RainEvent(object):
    #TAKE CARE: some measurements are every 1 hour instead of every 5 minutes
    #THerefore, check largest difference between start and end
    """"""
    duration = 0
    #type_rain = ""
    n_measurements = 0
    total_mm = 0
    amounts_mm = []
    rain_ID = 0
    indexes = []
    begin = 0
    end = 0
    #time_bf = 0 # Time before another rain event happens
    
    def __init__(self, begin, end, duration, n_measurements, total_mm, amounts_mm, rain_ID,
                 indexes):
        
        self.begin = begin
        self.end = end
        self.duration = duration
        #self.type_rain = type_rain
        self.n_measurements = n_measurements
        self.total_mm = total_mm
        self.amounts_mm = amounts_mm
        self.rain_ID = rain_ID
        self.indexes = indexes
        #self.time_bf = 0

        
        
        
    def get_df(self, flow_or_level_df):
        #Getting the event in a dataframe format with corresponding hours of flow
        #Linked to rain
        df = flow_or_level_df[flow_or_level_df["Begin"].dt.isin(self.dates)]
        df["Rain amounts"] = self.amounts_mm
        df["Begin"] = self.begin
        df["End"] = self.end
        
def event_parsing(df_events):
    events = []
    
    inst = set(df_events["rain_event_ID"])
    ids = [i for i in inst if i != "None"]
    
    for event_id in ids:
        relevant_df = df_events[df_events["rain_event_ID"] == event_id]
        begin = relevant_df["Begin"].iloc[0]
        end = relevant_df["End"].iloc[-1]
        duration = end - begin
        n_measurements = len(relevant_df)
        total_mm = sum(relevant_df.iloc[:, 0])
        amounts_mm = list(relevant_df.iloc[:, 0])
        indexes = list(relevant_df.index)
        
        events.append(RainEvent(begin, end, duration, n_measurements,
                                total_mm, amounts_mm, event_id, indexes))
    return events

#rain_df = streets_rain(station_names, path_linkinfo, path_rain)
#events_df = rain_events(station_names, rain_df)
#df = pd.concat([pd.read_csv(file) for file in glob.glob(path4+"/*.*")], ignore_index = True)
#events for haarsteeg
#events = event_parsing(events_df[0])

    