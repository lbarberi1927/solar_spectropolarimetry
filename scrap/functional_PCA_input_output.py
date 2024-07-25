import os

import numpy as np
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.misc.scoring import mean_absolute_percentage_error

from configs.NN_config import hparams
from configs.data import DATASET_PATH, DATA_FOLDER, FUNCTION_DOMAIN_POINTS
from src.utils import get_project_root, verify_path_exists

root = get_project_root()


def get_fPCA_scores(profile, data, grid_points, evaluate=True):
    n_components = hparams.FPCA.N_COMPONENTS
    if profile == "I":
        matrix = data[:, 5].reshape(-1, FUNCTION_DOMAIN_POINTS)
    elif profile == "Q":
        matrix = data[:, 6].reshape(-1, FUNCTION_DOMAIN_POINTS)
    elif profile == "U":
        matrix = data[:, 7].reshape(-1, FUNCTION_DOMAIN_POINTS)
    elif profile == "V":
        matrix = data[:, 8].reshape(-1, FUNCTION_DOMAIN_POINTS)

    print(f"computing fPCA for profile {profile}...")
    fdata = skfda.FDataGrid(data_matrix=matrix, grid_points=grid_points)
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fdata)

    fpca_params = {
        "components_": fpca.components_,
        "mean_": fpca.mean_,
        "singular_values": fpca.singular_values_,
    }

    if evaluate:
        evaluate_reconstructions(fpca, fdata, scores, profile)

    return scores, fpca_params


def evaluate_reconstructions(fpca, fdata, scores, profile):
    print("evaluating fPCA reconstructions for profile", profile)
    reconstructions = fpca.inverse_transform(scores)
    MAPE = mean_absolute_percentage_error(fdata, reconstructions)
    print(f"Mean Absolute Percentage Error for {profile}:", MAPE)



def main():
    # Load the data
    print("loading data...")
    data = np.loadtxt(os.path.join(root, DATASET_PATH), delimiter=",")
    grid_points = data[:FUNCTION_DOMAIN_POINTS, 4].reshape(1, -1)

    repetitions = int(len(data) / FUNCTION_DOMAIN_POINTS)
    input = list()

    for k in range(repetitions):
        input.append(data[k * FUNCTION_DOMAIN_POINTS, :4])

    input = np.array(input)
    save_folder = os.path.join(root, DATA_FOLDER)
    verify_path_exists(save_folder)
    print("saving variables...")
    np.savetxt(os.path.join(save_folder, "variables.csv"), input, delimiter=",")

    for profile in ["I", "Q", "U", "V"]:
        scores, params = get_fPCA_scores(profile, data, grid_points)
        save_folder = os.path.join(root, DATA_FOLDER, profile)
        verify_path_exists(save_folder)

        np.savetxt(
            os.path.join(save_folder, f"{profile}_scores.csv"), scores, delimiter=","
        )
        np.save(os.path.join(save_folder, f"{profile}_decomposition.npy"), params)

        print(f"saved {profile} scores")


if __name__ == "__main__":
    main()
