# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import glob

path = "C:/Users/maren/OneDrive/Dokumente/Muffin/Data Challenge 3/jbg060/data/sewer_data/data_pump/RG8150/RG8150"
path2 = "C:/Users/maren/OneDrive/Dokumente/Muffin/Data Challenge 3/jbg060/data/sewer_data/data_pump/RG8170/RG8170"

all_files = glob.glob(path + "/*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0, delimiter = ";")
    li.append(df)
comb50 = pd.concat(li, axis=0, ignore_index=True)
comb50["City.PumpType"], comb50["column_idk"], comb50["measurementType"] = comb50["Tagname"].str.split("/").str
comb50.drop(["Tagname"], inplace = True, axis=1)

#Saving to a separate csv only the file for pump RG8150
#comb50.to_csv("combined_csv50.csv")

all_files2 = glob.glob(path2 + "/*.csv")

li2 = []
for filename in all_files2:
    df = pd.read_csv(filename, index_col=None, header=0, delimiter = ";")
    li2.append(df)
    
comb70 = pd.concat(li2, axis=0, ignore_index=True)
comb70["City.PumpType"], comb70["column_idk"], comb70["measurementType"] = comb70["Tagname"].str.split("/").str
comb70.drop(["Tagname"], inplace = True, axis=1)


#comb70.to_csv("combined_csv70.csv")

both_pumps = pd.concat([comb50, comb70], axis=0, ignore_index=True)
both_pumps.to_csv("both_pumps.csv")
