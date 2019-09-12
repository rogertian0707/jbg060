# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 00:07:02 2019

@author: Hermii
"""
import pandas as pd
import seaborn as sns

path = "C:/Users/Hermii/Desktop/Data Challenge 3/repo/jbg060/code/"
#csv you get from temp.py
comb = pd.read_csv(path +"both_pumps.csv")

#Specify only "flow data" and 1 pump
flow = comb[(comb["measurementType"] == "Debietmeting.Q") & (comb["City.PumpType"] == "GBS_DB.RG8150")]
flow["TimeStamp"] = pd.to_datetime(flow["TimeStamp"])

flow["day"] = flow["TimeStamp"].dt.day_name()
flow["hour"] = flow["TimeStamp"].dt.hour

g = sns.FacetGrid(data = flow.groupby([
        "day", "hour"
]).hour.count().to_frame(name='day_hour_count').reset_index(), col='day', col_order=[
    'Sunday',
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday'
], col_wrap=3)

g.map(sns.barplot, "hour", "day_hour_count");

 
