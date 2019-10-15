# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:26:47 2019

@author: maren
"""
from random import shuffle
from multiprocessing import Process, Manager
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
 #The naive model still needs to be adjusted using the raingrid data to more accurately label days as dry or rainy

def worker(dry, station_names, n): #creating worker function to execute multiple processes simultaneously
    error = {}
    for pump, name in zip(dry, station_names):
        error[name] = []
        for month in pump['Month'].unique():
            #randomly select 10 percent of days as test data
            dat = pump[pump['Month'] == month]
            days = [i for i in range(1, max(dat['day_ofthe_month']))]
            shuffle(days)
            train = dat[~dat['day_ofthe_month'].isin(days[:(len(days)+1)//10])] 
            test = dat[dat['day_ofthe_month'].isin(days[:(len(days)+1)//10])] 
            #aggregate
            train = train.groupby(['Weekday', 'Hour']).mean()['flow']
            test = test.groupby(['Weekday', 'Hour']).mean()['flow']
            #if not existent: create pickleRick folder in data folder before running!
            with open("C:/Users/maren/OneDrive/Dokumente/Muffin/Data Challenge 3/jbg060/data/pickleRick/"+name+str(n)+month+'.dat', 'wb') as f:
                pickle.dump(abs((train-test).dropna()), f) #save absolute errors as pickle files

if __name__ == '__main__':
    station_names = ["Bokhoven", "Hertogenbosch (Helftheuvelweg)",
                     "Hertogenbosch (Rompert)", "Hertogenbosch (Oude Engelenseweg)",
                     "Hertogenbosch (Maasport)"]
    
    path = "../data/model_data/"
    dfs = [pd.read_csv(str(path) + str(filename)) for filename in os.listdir(path)]
    
    pumps = {name: dfs[i] for i, name in enumerate(station_names)} #dictionary containing all the dataframes
    
    Bok, Helft, Romp, OudEng, Maas = (pumps[i] for i in pumps.keys())
    
    data = [Bok, Helft, Romp, OudEng, Maas]   
    
    dry = [i[i['rain_N_ago']==0] for i in data] #filtering out rainy days
    
    
    
    
    now = time.time()
    n = 0
    for montecarlo in range(25): #starting 25*4 processes so 100 montecarlo simulations in total
        plist = []
        for i in range(4):
            plist.append(Process(target=worker, args=[dry, station_names, n])) 
            n += 1
            plist[i].start()
            
        for i in range(4):        
             plist[i].join()
             
    print("Time taken: ", time.time()-now)