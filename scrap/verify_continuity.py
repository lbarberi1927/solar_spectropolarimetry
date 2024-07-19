#%%
import matplotlib.pyplot as plt
import numpy as np
import os

from config import hparams, DATA_FOLDER
from src.utils import get_project_root

root = get_project_root()

all_data = np.loadtxt(os.path.join(root, DATA_FOLDER, "fe6302_basic.csv"), delimiter=",")
print("yo")

#%%
fix_dl = 0
fix_g = 6000
fix_G = 3500
fix_small_g = 0
fix_c = 22

dl_index = 4
g_index = 0
G_index = 1
small_g_index = 2
c_index = 3

def plot_continuity(variable, dep_var="I"):
    if variable=="g":
        study_index = g_index
        plot_data = all_data[all_data[:, dl_index] == fix_dl]
        plot_data = plot_data[plot_data[:, G_index] == fix_G]
        plot_data = plot_data[plot_data[:, small_g_index] == fix_small_g]
        plot_data = plot_data[plot_data[:, c_index] == fix_c]
    elif variable=="G":
        study_index = G_index
        plot_data = all_data[all_data[:, dl_index] == fix_dl]
        plot_data = plot_data[plot_data[:, g_index] == fix_g]
        plot_data = plot_data[plot_data[:, small_g_index] == fix_small_g]
        plot_data = plot_data[plot_data[:, c_index] == fix_c]
    elif variable=="small_g":
        study_index = small_g_index
        plot_data = all_data[all_data[:, dl_index] == fix_dl]
        plot_data = plot_data[plot_data[:, G_index] == fix_G]
        plot_data = plot_data[plot_data[:, g_index] == fix_g]
        plot_data = plot_data[plot_data[:, c_index] == fix_c]
    elif variable=="c":
        study_index = c_index
        plot_data = all_data[all_data[:, dl_index] == fix_dl]
        plot_data = plot_data[plot_data[:, G_index] == fix_G]
        plot_data = plot_data[plot_data[:, small_g_index] == fix_small_g]
        plot_data = plot_data[plot_data[:, g_index] == fix_g]

    if dep_var=="I":
        dep_var_index = 5
    elif dep_var=="U":
        dep_var_index = 6
    elif dep_var=="V":
        dep_var_index = 7
    elif dep_var=="Q":
        dep_var_index = 8

    plt.plot(plot_data[:, study_index], plot_data[:, dep_var_index], 'o')
    plt.title(f"{dep_var} vs. {variable}")
    plt.show()

#%%
dep_var = "Q"
plot_continuity("g", dep_var)
plot_continuity("G", dep_var)
plot_continuity("small_g", dep_var)
plot_continuity("c", dep_var)
