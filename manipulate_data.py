import numpy as np
import pandas as pd
import os
from tqdm import tqdm


# Path to dataset (wrong atm)
path_train_B = "C:/Users/Vesa/Documents/GitHub/Statistical-Genetics-and-Personalised-Medicine-Assignment/Data_sets/B"

train_B =0

# Loop through the files in the folder
for idx, filename in enumerate(tqdm(os.listdir(path_train_B))):
    if(filename.endswith(".psv")):
        
        # Patient id is the file name
        id = filename[1:-4]
        
        # Read one file
        patient = pd.read_csv((path_train_B + filename), sep='|')
        (r, c) = patient.shape
    
        # Add time and patient id as own columns
        time_list = patient.index
        patient["Time"] = time_list
        id_list = [id] * r
        patient["Id"] = id_list

        # Convert to numpy 
        patient_np = patient.to_numpy()

        # Concatinate data matrix with patients data
        if idx == 0:
            train_B = patient_np
        else:
            train_B = np.concatenate((train_B, patient_np), axis=0)


print(train_B.shape)
# Testing git 

