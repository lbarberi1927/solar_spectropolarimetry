import os
from pathlib import Path

import numpy as np
from os.path import expanduser

from configs.data import SIRIUS_DATA_ORIGIN_FOLDER, SIRIUS_DATA_DESTINATION_PATH
from src.utils import verify_path_exists

# Get the home directory of the current user
home = expanduser("~")

# Construct the path to the data folder
data_folder_path = os.path.join(home, "..", SIRIUS_DATA_ORIGIN_FOLDER)
os.chdir(data_folder_path)

# Base file path for data files
file_path = "fe6302_lte"

# Define temperature range and step
low_temp = 3600
high_temp = 6000
temp_step = 100
temperatures = np.arange(low_temp, high_temp + 1, temp_step)
g_conf = [str(t).zfill(4) for t in temperatures]

# Define magnetic strength range and step
low_mag_strength = 0
high_mag_strength = 5000
mag_strength_step = 200
mag_strengths = np.arange(low_mag_strength, high_mag_strength + 1, mag_strength_step)
G_conf = [str(g).zfill(4) for g in mag_strengths]

# Define gamma angles (custom range and steps)
low_gamma = 0
high_gamma = 180
gamma_step = 5
# gammas = np.arange(low_gamma, high_gamma + 1, gamma_step)
gammas = [0, 20, 45, 65, 90, 110, 135, 155, 180]
g_2_conf = [str(g).zfill(3) for g in gammas]

# Define xi angles range and step
low_xi = 0
high_xi = 180
xi_step = 11.25
xis = np.arange(low_xi, high_xi + 1, xi_step)
c_conf = [str(int(c)).zfill(3) for c in xis]

# Initialize a list to store all data
all_data = list()

# Loop through all combinations of parameters and load data
for g in g_conf:
    for G in G_conf:
        for g_2 in g_2_conf:
            for c in c_conf:
                file = file_path + g + "g45v1_" + G + "G_g" + g_2 + "_c" + c + ".dat"
                try:
                    # Load the data from the file, skipping the first two rows
                    data = np.loadtxt(file, skiprows=2)

                    # Create configuration array for the current file
                    configs = np.array([[int(g), int(G), int(g_2), int(c)]])
                    rep_configs = np.tile(configs, (data.shape[0], 1))

                    # Concatenate configurations with the data
                    new_data = np.concatenate([rep_configs, data], axis=1)
                    all_data.append(new_data)
                except:
                    print("File not found: ", file)

# Concatenate all loaded data
all_data = np.vstack(all_data)

# Save the concatenated data to the destination path
print(f"saving to {SIRIUS_DATA_DESTINATION_PATH} on sirius")
sink = os.path.join(home, SIRIUS_DATA_DESTINATION_PATH)
verify_path_exists(Path(sink).parent)
np.savetxt(sink, all_data, delimiter=",")
