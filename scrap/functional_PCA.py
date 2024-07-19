import os

import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.exploratory.visualization import FPCAPlot
import matplotlib.pyplot as plt

from config import DATA_FOLDER
from src.utils import get_project_root

root = get_project_root()

# Load the data
data = np.loadtxt(os.path.join(root, DATA_FOLDER, 'fe6302_lte6000g45v1_3500G_g180_c112.dat'), skiprows=2)
x = data[:, 0]
y = data[:, 1:].T
tiled = np.tile(x.reshape(1, -1), (4, 1))

fdata = skfda.FDataGrid(data_matrix=y, grid_points=x.reshape(1, -1))

fpca = FPCA(n_components=2)
fpca.fit(fdata)

fdata.plot()
fpca.components_.plot()

trans = fpca.transform(fdata)
inverse = fpca.inverse_transform(trans)
inverse.plot()
plt.show()
print("trans")
