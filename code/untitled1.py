# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:51:39 2019

@author: Hermii
"""
import pandas as pd
path4_linkinfo = "C:/Users/Hermii/Desktop/Data Challenge 3/repo/waterschap-aa-en-maas_sewage_2019/sewer_model"
print(pd.read_csv(path4_linkinfo+"/20180717_dump riodat rioleringsdeelgebieden_matched_to_rainfall_locations.csv").head(100))