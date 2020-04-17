# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:37 2020

@author: Jonne
"""

import numpy as np
import matplotlib.pyplot as plt

dict_B = np.load("E:/AALTO/Kevät2020/STATISTICAL GENETICS  & PERS. MED/Project/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/train_B.npz",
                 allow_pickle=True)
# Data B in numpy format 
#ata_B = dict_B['arr_0']

print(data_B.shape)
# Sepsislabel = 41(40) ; Heartrate = 1

# Target is to select:
#   The vital of people who had sepsis 
#   Same amount of patients that did not have sepsis

# Vital signs
#   HR = 0
#   O2sat = 1
#   Temp = 2
#   SBP = 3
#   MAP = 4
#   DBP = 5
#   Resp = 6
#   EtCO2 = 7
vital_sign = 7

target_rows = data_B[data_B[:,40]==1]
ids = np.unique(target_rows[:,42])

sum = np.ones(10)
skipped = 0
counted = 0

# Loop through everyone that has sepsis and calculate the average vital sign difference from patients base value
for idx, identity in enumerate(ids):
    roi = data_B[data_B[:,42]==identity]
    time_of_sepsis = roi[roi[:,40]==1][0,41] 
    
    if(time_of_sepsis < 5):
        skipped+=1
        continue
    elif(roi.shape[0] < time_of_sepsis+5):
        skipped+=1
        continue
    
    low = time_of_sepsis-5
    up = time_of_sepsis+5
    base_before = np.nanmean(roi[0:time_of_sepsis,vital_sign].astype(float))
    base_after = np.nanmean(roi[time_of_sepsis:,vital_sign].astype(float))
    
    # Check if base values are nan -> some patient has all vital signs nans after sepsis
    if np.isnan(base_before) or np.isnan(base_after):
        skipped+=1
        continue
    
    measured_low = roi[low:time_of_sepsis,vital_sign].astype(float)
    measured_high = roi[time_of_sepsis:up,vital_sign].astype(float)
    
    # Replace nan values with base
        # Base_after used only when replacing nan values 
    temp_low = np.isnan(measured_low) 
    temp_high = np.isnan(measured_high)
    measured_low[temp_low] = base_before
    measured_high[temp_high] = base_after
    
    # Join the before and after sepsis measurements
    measured = np.concatenate((measured_low, measured_high))
    
    # Divide all instances by the pre-sepsis basevalue
    measured = measured/base_before
    
    # Stack matrix with all values 
    if idx==0:
        all_measured = np.expand_dims(measured, axis=0)
    else:
        all_measured = np.concatenate((all_measured, np.expand_dims(measured, axis=0)), axis=0)
    
# Calculate the mean over all looped patients
print(all_measured.shape)
avg_measures = np.mean(all_measured, axis=0)

print("The average values to plot are " + str(avg_measures))
print("Skipped values " + str(skipped))
amount_counted = idx-skipped+1
print("Used calculating the average " + str(amount_counted))


# Create a [1,10] vector of "clean" patients data to compare 
reference_patients = data_B[:,42]
ids_all = np.unique(reference_patients)

# Loop through amount_counted times to get the avg for reference patients
loop = 0
while(loop <= amount_counted):
    patient = np.random.choice(ids_all,1)
    # Check if the patient is sepsis patient
    if(np.isin(patient,ids)):
        continue
    else:
        roi = data_B[data_B[:,42]==patient]

        # Skip patients without enough datapoints        
        if(roi.shape[0] < 15):
            continue
        
        base = np.nanmean(roi[:,vital_sign].astype(float))
        # Check if every vital value has been nan -> skip
        if np.isnan(base):
            continue
        
        # Select 10 values after 5 hours (time for patient to rest before)
        measured = roi[5:15,vital_sign].astype(float)
        
        # Replace nan values
        temp = np.isnan(measured) 
        measured[temp] = base
        
        # Divide all instances by the pre-sepsis basevalue
        measured = measured/base
        
             # Stack matrix with all values 
        if loop==0:
            all_measured_healthy = np.expand_dims(measured, axis=0)
        else:
            all_measured_healthy = np.concatenate((all_measured_healthy, np.expand_dims(measured, axis=0)), axis=0)
            

    loop+=1
    
avg_measures_healthy = np.mean(all_measured_healthy, axis=0)

print("The average values of healthy patient are " + str(avg_measures_healthy))
print("Used calculating the healthy average " + str(loop))


# PROBLEM : only takes account those that have nice 5+5h period in data -> wasting data
    # Also should the nan values be changed into something else  


plt.plot(avg_measures)
plt.plot(avg_measures_healthy)
plt.axvspan(5, 10, color='red', alpha=0.3)





