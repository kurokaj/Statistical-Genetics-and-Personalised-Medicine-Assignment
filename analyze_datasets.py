# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import time


# Path to dataset (wrong atm)
dataset_dir = "C:/Users/Vesa/Documents/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/"

# Load dataset archives

print("Select dataset ([a]/b):")
dataset_selection = input().lower()


if dataset_selection == 'a':
    print("Dataset A selected, loading..\n")
    df = np.load(dataset_dir + "train_A.npz", allow_pickle=1)
    df = df['arr_0']
    df = pd.DataFrame(df)
    df_sepsishours = np.load(dataset_dir + "df_sepsishours_A.npy", allow_pickle=1)
    df_sepsishours = pd.DataFrame(df_sepsishours)
    print("Dataset A successfully loaded.\n")
elif dataset_selection == 'b':
    print("Dataset B selected, loading..\n")
    df = np.load(dataset_dir + "train_B.npz", allow_pickle=1)
    df = df['arr_0']
    df = pd.DataFrame(df)
    df_sepsishours = np.load(dataset_dir + "df_sepsishours_B.npy", allow_pickle=1)
    df_sepsishours = pd.DataFrame(df_sepsishours)
    print("Dataset B successfully loaded.\n")
else:
    print("No valid dataset selected, defaulting to A...\n")
    df = np.load(dataset_dir + "train_A.npz", allow_pickle=1)
    df = df['arr_0']
    df = pd.DataFrame(df)
    df_sepsishours = np.load(dataset_dir + "df_sepsishours_A.npy", allow_pickle=1)
    df_sepsishours = pd.DataFrame(df_sepsishours)
    print("Dataset A successfully loaded.\n")



# Enter column names

df.columns = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess",
            "HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos",
            "Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose",
            "Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
            "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age",
            "Gender","Unit1","Unit2","HospAdmTime","ICULOS","SepsisLabel",
            "HoursInICU","ID"]

# Create a mini version of the dataframe for visual inspection 
# (the original dataframe is too big to open with 700k+ lines)

df_mini = pd.DataFrame(df.iloc[:10000])


# Start working on dataset 

#######################################################################
#######################################################################
#######################################################################

# # 1. Count how many hours each patient has with, and without, sepsis

# # Get unique values in "ID" column to identify patients
# unique_ids = df.ID.unique().tolist()
# hours = [0] * len(unique_ids)
# sepsishours_columns = [unique_ids, hours, hours]

# df_sepsishours = pd.DataFrame(sepsishours_columns)
# df_sepsishours = df_sepsishours.transpose()
# df_sepsishours.columns = ["ID","HoursWithSepsis","HoursWithoutSepsis"]

# # Clear unneeded variables from variable explorer

# del hours,sepsishours_columns

# # Construct the dataset (Note: This for loop takes 1+ hour!) 
# # Uncomment only if needed!

# for index, row in tqdm(df.iterrows()):
#     patientID = row['ID']
#     if row['SepsisLabel'] == 1:
#         currentSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"]
#         df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"] = currentSepsisHours + 1
#     elif row['SepsisLabel'] == 0:
#         currentNoSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"]
#         df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"] = currentNoSepsisHours + 1

#######################################################################
#######################################################################
#######################################################################
        
# 2. Create lists of patients with, and without, sepsis

# Initiate arrays

patients_with_sepsis = []
patients_without_sepsis = []

        
# Populate arrays

print("\nPopulating lists patients_with_sepsis and patients_without_sepsis..\n")
time.sleep(0.1)
for index, row in tqdm(df_sepsishours.iterrows()):
    if row[1] == 0:
        patients_without_sepsis.append(row[0])
    else:
        patients_with_sepsis.append(row[0])
time.sleep(0.1)
del index,row
print("\nLists populated.\n")
        
        
# Convert to numpy array for saving etc.; uncomment if needed

# patients_with_sepsis_np = np.asarray(patients_with_sepsis)
# patients_without_sepsis_np = np.asarray(patients_without_sepsis)


#######################################################################
#######################################################################
#######################################################################

# 3. Create additional column for "TimeToSepsis"
# The first sepsis hour label will have the value "0", ones before it
# will have values -1, -2, -3.. depending on how many hours until sepsis,
# ones after it will have values 1, 2, 3.. depending on how many hours after.

# Note: This is only applied to the main dataframe, which you can select
# from the beginning. It's not applied to both A and B!

df['TimeToSepsis'] = np.nan

df_mini_orig = pd.DataFrame(df.iloc[:10000])

for patient in patients_with_sepsis:
    df_mini.loc[df_mini.ID == patient, "TimeToSepsis"] = 1
    
