import os

import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA

from configs.NN_config import hparams, DATA_FOLDER
from src.utils import get_project_root

root = get_project_root()

def get_fPCA_scores(profile, data, grid_points):
    n_components = hparams.FPCA.N_COMPONENTS
    if profile=="I":
        matrix = data[:, 5].reshape(-1, 301)
    elif profile=="Q":
        matrix = data[:, 6].reshape(-1, 301)
    elif profile=="U":
        matrix = data[:, 7].reshape(-1, 301)
    elif profile=="V":
        matrix = data[:, 8].reshape(-1, 301)

    fdata = skfda.FDataGrid(data_matrix=matrix, grid_points=grid_points)
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fdata)

    fpca_params = {
        "components_": fpca.components_,
        "mean_": fpca.mean_,
        "singular_values": fpca.singular_values_,
    }

    return scores, fpca_params


def main():
    # Load the data
    data = np.loadtxt(os.path.join(root, DATA_FOLDER, 'fe6302_basic.csv'), delimiter=",")
    grid_points = data[:301, 4].reshape(1, -1)

    repetitions = int(len(data)/301)
    input = list()

    for k in range(repetitions):
        input.append(data[k*301, :4])

    input = np.array(input)
    np.savetxt(os.path.join(root, DATA_FOLDER, "variables.csv"), input, delimiter=",")

    for profile in ["I", "Q", "U", "V"]:
        scores, params = get_fPCA_scores(profile, data, grid_points)
        np.savetxt(os.path.join(root, DATA_FOLDER, f"{profile}_scores.csv"), scores, delimiter=",")
        np.save(os.path.join(root, DATA_FOLDER, f"{profile}_decomposition.npy"), params)
        # to read: read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
        print(f"saved {profile} scores")


if __name__ == '__main__':
    main()
