# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt




# Path to dataset (wrong atm)
dataset_dir = "R:/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/"
graph_dir = "R:/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Graphs/"

# Load dataset archives


df = np.load(dataset_dir + "df_all.npz", allow_pickle=1)
df = df['arr_0']
df = pd.DataFrame(df)
print("Dataset successfully loaded.\n")

df_mini = pd.DataFrame(df.iloc[:10000])


# Enter column names

df.columns = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2","BaseExcess",
            "HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN","Alkalinephos",
            "Calcium","Chloride","Creatinine","Bilirubin_direct","Glucose",
            "Lactate","Magnesium","Phosphate","Potassium","Bilirubin_total",
            "TroponinI","Hct","Hgb","PTT","WBC","Fibrinogen","Platelets","Age",
            "Gender","Unit1","Unit2","HospAdmTime","ICULOS","SepsisLabel",
            "HoursInICU","ID"]

#######################################################################
#######################################################################
#######################################################################

# # Start working on dataset 

# size_sepsislabel1 = len(df.loc[df['SepsisLabel'] == 1]['Temp'])
# size_sepsislabel0 = len(df.loc[df['SepsisLabel'] == 0]['Temp'])

# size = min(size_sepsislabel1, size_sepsislabel0)

# del size_sepsislabel1,size_sepsislabel0


# plt.hist(df_mini.loc[df_mini['SepsisLabel'] == 1]['Temp'], bins=50, label = "Temp Patients", alpha=0.5)
# plt.hist(df_mini.loc[df_mini['SepsisLabel'] == 0]['Temp'], bins=50, label = "Temp Others", alpha=0.5)
# plt.legend(loc='best')
# plt.show()

# plt.hist(df_temp, bins=50, label = "Temp", alpha=0.5)
# plt.hist(df_temp2, bins=50, label = "Temp2", alpha=0.5)
# plt.legend(loc='best')

#######################################################################
#######################################################################
#######################################################################

# Print all graphs in their own file

    
for col in tqdm(range(0, len(df.columns) - 8)):
    size_sepsislabel1 = len(df.loc[df['SepsisLabel'] == 1][label].dropna())
    size_sepsislabel0 = len(df.loc[df['SepsisLabel'] == 0][label].dropna())
    size = min(size_sepsislabel1, size_sepsislabel0)
    del size_sepsislabel1,size_sepsislabel0
    if size == 0:
        print('Skipping ' + label + ' due to excess NaN values.')
    else:        
        data_sepsislabel1 = df.loc[df['SepsisLabel'] == 1][label].dropna().iloc[:size]
        data_sepsislabel0 = df.loc[df['SepsisLabel'] == 0][label].dropna().iloc[:size]
        plt.title(label + " distributions, n = " + str(size) + " for each group")
        plt.hist(data_sepsislabel1, bins=50, label = label + ", SepsisLabel 1", alpha=0.5)
        plt.hist(data_sepsislabel0, bins=50, label = label + ", SepsisLabel 0", alpha=0.5)
        plt.legend(loc='best')
        plt.savefig(graph_dir + str(col) + '_' + label + ".png")
        plt.clf()

