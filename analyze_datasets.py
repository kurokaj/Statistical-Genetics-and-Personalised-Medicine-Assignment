# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import time



# Path to dataset (wrong atm)
dataset_dir = "R:/GitHub/Data_sets/"

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
    dataset_selection = 'a'
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
            "HoursInICU","ID"]#, "TimeToSepsis"]

#df_sepsishours['3'] = df_sepsishours[1] +df_sepsishours[2] #
df_sepsishours.columns = ["ID",
                          "Hours with sepsis", 
                          "Hours without sepsis",
                          "Total hours",
                          "First row in the sheet"]


 

# Create a mini version of the dataframe for visual inspection 
# (the original dataframe is too big to open with 700k+ lines)

df_mini = pd.DataFrame(df.iloc[:10000])


# Start working on dataset 

#######################################################################
#######################################################################
#######################################################################

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

# Construct the dataset (Note: This for loop takes 1+ hour!) 
# Uncomment only if needed!

for index, row in tqdm(df.iterrows()):
    patientID = row['ID']
    if row['SepsisLabel'] == 1:
        currentSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"]
        df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithSepsis"] = currentSepsisHours + 1
    elif row['SepsisLabel'] == 0:
        currentNoSepsisHours = df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"]
        df_sepsishours.loc[df_sepsishours["ID"] == patientID, "HoursWithoutSepsis"] = currentNoSepsisHours + 1


# Calculate first occurrences of each patient based on number of hours(rows)

df_sepsishours['First Occurrence'] = 0 # Initialize column as zero

print("\nStep 1: Populating First Occurence -column..\n")
time.sleep(0.2)

for i in tqdm(range(len(df_sepsishours)-1)): # Start from second row
    df_sepsishours.iloc[i+1,3] = df_sepsishours.iloc[(i),3] + df_sepsishours.iloc[i,2]
 
time.sleep(0.2)
print("\nList populated.\n")

# Convert to Excel & numpy array for saving etc.; uncomment if needed
# Note: Make sure the output file has the correct filename!

if dataset_selection == 'a':
    df_sepsishours.to_excel(dataset_dir + "df_sepsishours_A.xlsx")
elif dataset_selection == 'b':
    df_sepsishours.to_excel(dataset_dir + "df_sepsishours_B.xlsx")
else:
    df_sepsishours.to_excel(dataset_dir + "df_sepsishours_A.xlsx")
print('df_sepsishours saved to Excel file.')

# df_sepsishours_np = np.asarray(df_sepsishours)

#######################################################################
#######################################################################
#######################################################################
        
# 2. Create lists of patients with, and without, sepsis

# Initiate arrays

patients_with_sepsis = []
patients_without_sepsis = []

        
# Populate arrays

print("\nStep 2: Populating lists patients_with_sepsis and patients_without_sepsis..\n")
time.sleep(0.1)
for index, row in tqdm(df_sepsishours.iterrows()):
    if row[1] == 0:
        patients_without_sepsis.append(row[0])
    else:
        patients_with_sepsis.append(row[0])
time.sleep(0.1)
del index,row
print("\npatients_with_sepsis and patients_without_sepsis lists populated.\n")
        
        
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
# The "TimeToSepsis" value is 0 on the last row before SepsisLabel 1.
# Patients who never experience sepsis will have TimeToSepsis 999.


# df_mini_orig = pd.DataFrame(df.iloc[:10000])

print("\nStep 3: Populating Time To Sepsis -column..\n")
time.sleep(0.2)

df['TimeToSepsis'] = 999
for patient in tqdm(patients_with_sepsis):
    index = df_sepsishours.loc[df_sepsishours['ID'] == patient]['First Occurrence'].item()
    totalHours = df_sepsishours.loc[df_sepsishours['ID'] == patient,'Total hours'].item()
    sepsisHours = df_sepsishours.loc[df_sepsishours['ID'] == patient,'Hours with sepsis'].item()
    for row in range(0,totalHours):
        df.loc[index+row,'TimeToSepsis'] = sepsisHours - totalHours + row + 1
    
time.sleep(0.2)
print("\nTime To Sepsis -column populated.\n")



#######################################################################
#######################################################################
#######################################################################

# 4. Additional modifications to the dataframe

# Replace SepsisLabel column with 1 for all sepsis patients

print("\nStep 4: Converting SepsisLabel -column to 1 for all sepsis patients..\n")
time.sleep(0.2)

for patient in tqdm(patients_with_sepsis):
    index = df_sepsishours.loc[df_sepsishours['ID'] == patient]['First Occurrence'].item()
    totalHours = df_sepsishours.loc[df_sepsishours['ID'] == patient,'Total hours'].item()
    sepsisHours = df_sepsishours.loc[df_sepsishours['ID'] == patient,'Hours with sepsis'].item()
    for row in range(0,totalHours):
        df.loc[index+row,'SepsisLabel'] = 1
        
time.sleep(0.2)
print("\nConversion done.\n")


# Optional: Convert to Excel & numpy array for saving etc.; uncomment if needed
# Note: Make sure the output file has the correct filename!

# df_np = np.asarray(df)
# df.to_excel("C:/Users/Vesa/Documents/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/df_A.xlsx")
# df.to_excel("C:/Users/Vesa/Documents/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/df_B.xlsx")



# Optional: Save dataframe to npz file, uncomment if needed

from numpy import savez_compressed
df_np = df.to_numpy()

print('Saving dataset..')

if dataset_selection == 'a':
    savedir = dataset_dir + 'df_A_modified.npz'
    savez_compressed(savedir, df_np)
elif dataset_selection == 'b':
    savedir = dataset_dir + 'df_B_modified.npz'
    savez_compressed(savedir, df_np)
else:
    savedir = dataset_dir + 'df_A_modified.npz'
    savez_compressed(savedir, df_np)
print('Dataset saved to ' + savedir + '.')





#######################################################################
#######################################################################
#######################################################################

# 5. Draw graphs based on the dataset

df_valueframe = pd.DataFrame()

# Select column names from the dataname as index

df_valueframe['Index'] = df.columns
df_valueframe['Mean (all)'] = 0
df_valueframe['Variance (all)'] = 0

for i in range(-5,5):
    df_valueframe['Mean ' + str(i)] = 0
    df_valueframe['Var ' + str(i)] = 0

print('\nPopulating df_valueframe dataframe..\n')

for row in tqdm(range(len(df_valueframe))):
    if df_valueframe['Index'][row] == 'ID':
        df_valueframe.loc[row,'Mean (all)'] = 0
        df_valueframe.loc[row,'Variance (all)'] = 0
    else:
        df_valueframe.loc[row,'Mean (all)'] = df[df_valueframe['Index'][row]].mean()
        df_valueframe.loc[row,'Variance (all)'] = df[df_valueframe['Index'][row]].var()


for row in tqdm(range(len(df_valueframe))):
    for i in range(-5,5):
        if df_valueframe['Index'][row] == 'ID':
            df_valueframe.loc[row,'Mean ' + str(i)] = 0
            df_valueframe.loc[row,'Var ' + str(i)] = 0
        else:
            df_valueframe.loc[row,'Mean ' + str(i)] = df[df.TimeToSepsis == i][df_valueframe['Index'][row]].mean()
            df_valueframe.loc[row,'Var ' + str(i)] = df[df.TimeToSepsis == i][df_valueframe['Index'][row]].var()
    

print('\nDataframe populated.\n')





#################

print("Script ended. The dataset used was: " + dataset_selection + ".")

