# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# Path to dataset (wrong atm)
dataset_dir = "C:/Users/Vesa/Documents/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/"

# Load dataset archives

dataset_a = np.load(dataset_dir + "train_A.npz", allow_pickle=1)
dataset_b = np.load(dataset_dir + "train_B.npz", allow_pickle=1)

# Extract arrays from archives

dataset_a = dataset_a['arr_0']
dataset_b = dataset_b['arr_0']

# NOTE! Comment/uncomment the below lines to switch the main dataframe contents!
# Select which dataset to work on (they're the same kind of data)

#df = dataset_a
df = dataset_b

# Convert to Pandas dataframe

df = pd.DataFrame(df)

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

df_mini = df.iloc[:10000]


# Delete unused datasets

del dataset_a,dataset_b


# Start working on dataset 

# 1. Count how many hours each patient has with, and without, sepsis

# Get unique values in "ID" column to identify patients
unique_ids = df.ID.unique().tolist()
hours = [0] * len(unique_ids)
sepsishours_columns = [unique_ids, hours, hours]

df_sepsishours = pd.DataFrame(sepsishours_columns)
df_sepsishours = df_sepsishours.transpose()
df_sepsishours.columns = ["ID","HoursWithSepsis","HoursWithoutSepsis"]

# Clear unneeded variables from variable explorer

del hours,sepsishours_columns

# 

for index, row in tqdm(df.iterrows()):
    patientID = row['ID']
    if row['SepsisLabel'] == 1:
        currentSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"]
        df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"] = currentSepsisHours + 1
    elif row['SepsisLabel'] == 0:
        currentNoSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"]
        df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"] = currentNoSepsisHours + 1
        

        

