import os

import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
import matplotlib.pyplot as plt

from configs.data import DATASET_PATH
from src.utils import get_project_root

root = get_project_root()

# Load the data
data = np.loadtxt(DATASET_PATH, delimiter=",")

matrix = data[:, 5].reshape(-1, 301)
grid_points = data[:301, 4].reshape(1, -1)

fdata = skfda.FDataGrid(data_matrix=matrix, grid_points=grid_points)

fpca = FPCA(n_components=2)
fpca.fit(fdata)

big_fpca = FPCA(n_components=14)
big_fpca.fit(fdata)

print("explained variance ratio with 2 PCs: ", fpca.explained_variance_ratio_)
print("explained variance ratio with 14 PCs: ", big_fpca.explained_variance_ratio_)

fdata.plot()
plt.xlabel('Δλ')
plt.ylabel('I')
plt.title("All Stokes I Profiles")
plt.savefig("../images/all_profiles.png")

fpca.components_.plot()
plt.xlabel('Δλ')
plt.ylabel('I')
plt.title("2 Principal Components")
plt.savefig("../images/2PCs.png")

big_fpca.components_.plot()
plt.title("14 Principal Components")
plt.xlabel('Δλ')
plt.ylabel('I')
plt.savefig("../images/14PCs.png")

#----- take example plot ----
index = np.random.randint(0, len(matrix))
print(index)
eg_data = matrix[index]
fig, ax = plt.subplots()

eg = skfda.FDataGrid(data_matrix=eg_data, grid_points=grid_points)
eg.plot()
plt.xlabel('Δλ')
plt.ylabel('I')
plt.title("Stokes I Profile")
plt.savefig("../images/example_profile.png")
eg.plot(axes=ax, color="blue", label="Observation")

trans = fpca.transform(eg)
inverse = fpca.inverse_transform(trans)

big_trans = big_fpca.transform(eg)
big_inverse = big_fpca.inverse_transform(big_trans)

print("coefficients for example function: ", big_trans)

inverse.plot(axes=ax, color="green", linestyle="dashed", label="2 PCs Reconstruction")
big_inverse.plot(axes=ax, color="red", linestyle="dashed", label="14 PCs Reconstruction")

ax.set_title(f"Principal Components Reconstruction")
ax.legend()
ax.set_xlabel('Δλ')
ax.set_ylabel('I')
fig.savefig("../images/PC_reconstruction.png")

plt.show()
