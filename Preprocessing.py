# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:01:29 2016

@author: Umair Ahmed
"""

#important imports
import pandas as pd
import math
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools
import csv


#Functions for calculating the Magnitude of any axis of Raw sensor Data
def magnitude(user_id):
    x2 = user_id['xAxis'] * user_id['xAxis']
    y2 = user_id['yAxis'] * user_id['yAxis']
    z2 = user_id['zAxis'] * user_id['zAxis']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m
    
    
def calc_magnitudes():
    for i in range (1,19):
        user_list[i-1]['magnitude'] = magnitude(user_list[i-1])
        

#Function for defining the window on data
def window(axis,dx=100):
    start = 0;
    size = axis.count();

    while (start < size):
        end = start + dx
        yield start,end
        start = start+int (dx/2)
        


#Features which are extracted from Raw sensor data
def window_summary(axis, start, end):
    acf = stattools.acf(axis[start:end])
    acv = stattools.acovf(axis[start:end])
    sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
    return [
        axis[start:end].mean(),
        axis[start:end].std(),
        axis[start:end].var(),
        axis[start:end].min(),
        axis[start:end].max(),
        acf.mean(), # mean auto correlation
        acf.std(), # standard deviation auto correlation
        acv.mean(), # mean auto covariance
        acv.std(), # standard deviation auto covariance
        skew(axis[start:end]),
        kurtosis(axis[start:end]),
        math.sqrt(sqd_error.mean())
    ]

def features(user_id):
    for (start, end) in window(user_id['timestamp']):
        features = []
        for axis in ['xAxis', 'yAxis', 'zAxis', 'magnitude']:
            features += window_summary(user_id[axis], start, end)
        yield features        

   
     

#Main code for Pre-processing of the Data
COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
user_list = []
titles_list=[]
user_to_auth = 0

for i in range (1,19):
    file_path = 'Dataset/'+str(i)+'.csv'
    user_list.append((pd.read_csv(file_path,header=None,names=COLUMNS))[:1100])

#Add an additional axis of magnitude of the sensor data
calc_magnitudes() 

#Write the feature vectors to a separate excel file
with open('Features/Features.csv', 'w') as out:
    rows = csv.writer(out)
    for i in range(0, len(user_list)):
        for f in features(user_list[i]):
            rows.writerow([i]+f)
                













