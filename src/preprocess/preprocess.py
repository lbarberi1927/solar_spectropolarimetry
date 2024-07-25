import os

import joblib
import numpy as np
import skfda
from skfda.misc.scoring import mean_squared_error
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.NN_config import hparams
from configs.data import DATASET_PATH, FUNCTION_DOMAIN_POINTS, DATA_FOLDER
from src.utils import get_project_root, verify_path_exists

root = get_project_root()


def create_variables(data):
    print("creating variables...")
    repetitions = int(len(data) / FUNCTION_DOMAIN_POINTS)
    x = list()
    y = list()

    for k in range(repetitions):
        x.append(data[k * FUNCTION_DOMAIN_POINTS, :4])
        y.append(data[k * FUNCTION_DOMAIN_POINTS: (k + 1) * FUNCTION_DOMAIN_POINTS, 5:9])

    x = np.array(x)
    y = np.array(y)

    return x, y


def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=hparams.TRAIN.TEST_SIZE,
        random_state=hparams.SEED,
    )
    print(
        "separated data: training - ",
        x_train.shape,
        y_train.shape,
        "testing - ",
        x_test.shape,
        y_test.shape,
    )

    return x_train, x_test, y_train, y_test


def get_fPCA_scores(profile, y_train, y_test, grid_points, evaluate=True):
    n_components = hparams.FPCA.N_COMPONENTS
    train_matrix = separate_response(y_train, profile)
    test_matrix = separate_response(y_test, profile)

    print(f"computing fPCA for profile {profile}...")
    fdata_train = skfda.FDataGrid(data_matrix=train_matrix, grid_points=grid_points)
    fdata_test = skfda.FDataGrid(data_matrix=test_matrix, grid_points=grid_points)

    fpca = FPCA(n_components=n_components)
    train_scores = fpca.fit_transform(fdata_train)
    test_scores = fpca.transform(fdata_test)

    fpca_params = {
        "components_": fpca.components_,
        "mean_": fpca.mean_,
        "singular_values": fpca.singular_values_,
    }

    if evaluate:
        evaluate_reconstructions(fpca, fdata_test, test_scores, profile)

    return train_scores, test_scores, fpca_params


def evaluate_reconstructions(fpca, fdata, scores, profile):
    print("evaluating fPCA reconstructions for profile", profile)
    reconstructions = fpca.inverse_transform(scores)
    MSE = mean_squared_error(fdata, reconstructions)
    print(f"Mean Squared Error for {profile}:", MSE)


def separate_response(data, profile):
    if profile == "I":
        matrix = data[:, :, 0]
    elif profile == "Q":
        matrix = data[:, :, 1]
    elif profile == "U":
        matrix = data[:, :, 2]
    elif profile == "V":
        matrix = data[:, :, 3]

    return matrix


def main():

    # Load the data
    print("loading data...")
    data = np.loadtxt(os.path.join(root, DATASET_PATH), delimiter=",")
    grid_points = data[:FUNCTION_DOMAIN_POINTS, 4].reshape(1, -1)

    x, y = create_variables(data)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    joblib.dump(scaler, os.path.join(root, DATA_FOLDER, "scaler.pkl"))

    x_train, x_test, y_obs_train, y_obs_test = split(x, y)

    print("saving variables...")
    save_folder = os.path.join(root, DATA_FOLDER)
    verify_path_exists(save_folder)
    np.savetxt(os.path.join(save_folder, "x_train.csv"), x_train, delimiter=",")
    np.savetxt(os.path.join(save_folder, "x_test.csv"), x_test, delimiter=",")
    np.save(os.path.join(save_folder, "y_obs_train.npy"), y_obs_train)
    np.save(os.path.join(save_folder, "y_obs_test.npy"), y_obs_test)
    np.savetxt(os.path.join(save_folder, "grid_points.csv"), grid_points, delimiter=",")
    print("saved")
    """
    y_obs_train = np.load(os.path.join(root, DATA_FOLDER, "y_obs_train.npy"))
    y_obs_test = np.load(os.path.join(root, DATA_FOLDER, "y_obs_test.npy"))
    grid_points = np.loadtxt(os.path.join(root, DATA_FOLDER, "grid_points.csv"), delimiter=",")
    """
    for profile in ["I", "Q", "U", "V"]:
        train_scores, test_scores, params = get_fPCA_scores(profile, y_obs_train, y_obs_test, grid_points)
        prof_obs_test = separate_response(y_obs_test, profile)

        print(f"saving {profile} scores...")
        save_folder = os.path.join(root, DATA_FOLDER, profile)
        verify_path_exists(save_folder)
        np.savetxt(
            os.path.join(save_folder, f"y_train.csv"), train_scores, delimiter=","
        )
        np.savetxt(
            os.path.join(save_folder, f"y_test.csv"), test_scores, delimiter=","
        )
        np.savetxt(
            os.path.join(save_folder, f"y_obs_test.csv"), prof_obs_test, delimiter=","
        )
        np.save(os.path.join(save_folder, f"{profile}_decomposition.npy"), params)


if __name__ == "__main__":
    main()
