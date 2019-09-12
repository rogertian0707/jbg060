# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:21:02 2019

@author: Hermii
"""
import glob
import pandas as pd

## Haarsteeg


path4_linkinfo = "C:/Users/Hermii/Desktop/Data Challenge 3/repo/waterschap-aa-en-maas_sewage_2019/sewer_model"
linkage = pd.read_csv(path4_linkinfo+"/20180717_dump riodat rioleringsdeelgebieden_matched_to_rainfall_locations.csv", header = 9)
street_names = list(linkage[linkage["Naam kern"] == "Haarsteeg"]["Naam / lokatie"])
street_names_refurbished = []
street_names.append("Begin")
street_names.append("Eind")

for i in street_names:
    k = i.find("(")
    street_names_refurbished.append(i[:k])
    
path3 = "C:/Users/Hermii/Desktop/Data Challenge 3/waterschap-aa-en-maas_sewage_2019/sewer_data/rain_timeseries"
all_files3 = glob.glob(path3 + "/*.*")
li3=[]
for filename in all_files3:
    df = pd.read_csv(filename, header=2)
    li3.append(df)
    
comb_rain = pd.concat(li3, axis=0,ignore_index=True)

#Here we have the timestamps and the rain fallen in station Haarsteeg
#I will write code for the other 6 stations tomorrow around lunch
#Here you only need to add the pump data on RG8170 which corresponds to Haarsteeg (which I believe matches the shape
#of this resulting variable areas_Haarsteeg_pumpstation)


areas_Haarsteeg_pumpstation = comb_rain[street_names]

