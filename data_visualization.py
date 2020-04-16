# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:37 2020

@author: Jonne
"""

import numpy as np


dict_B = np.load("E:/AALTO/Kevät2020/STATISTICAL GENETICS  & PERS. MED/Project/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/train_B.npz",
                 allow_pickle=True)
# Data B in numpy format 
data_B = dict_B['arr_0']

print(data_B.shape)
# Sepsislabel = 41(40) ; Heartrate = 1

# 
# Target is to select:
#   The heart rate of people who had sepsis 
#   Time instace they had it
#   ID
  
target_rows = data_B[data_B[:,40]==1]
ids = np.unique(target_rows[:,42])

sum = np.ones(10)
skipped = 0
counted = 0

# Loop through everyone that has sepsis
for idx, identity in enumerate(ids):
    roi = data_B[data_B[:,42]==identity]
    time_of_sepsis = roi[roi[:,40]==1][0,41] 
    
    if(time_of_sepsis < 5):
        skipped+=1
        continue
    elif(roi.shape[0] < 5):
        skipped+=1
        continue
    
    low = time_of_sepsis-5
    up = time_of_sepsis+5
    base_before = np.nanmean(roi[0:time_of_sepsis,0])
    base_after = np.nanmean(roi[time_of_sepsis:,0])
    
    measured_low = roi[low:time_of_sepsis,0]
    measured_high = roi[time_of_sepsis:up,0]
    
    # Replace nan values with base
    measured_low = np.nan_to_num(measured_low, nan=base_before)
    measured_high = np.nan_to_num(measured_high, nan=base_after)

    measured = np.concatenate(measured_low, measured_high)
    
    # Calculate the mean over all looped patients
    

# PROBLEM : only takes account those that have nice 5+5h period in data -> wasting data
    # Also should the nan values be changed into something else

avg = sum/counted
print(avg)

