import os

import joblib
import numpy as np
import skfda
from skfda.misc.scoring import mean_squared_error
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from configs.NN_config import hparams, MODEL_SAVE_FOLDER
from configs.data import DATASET_PATH, FUNCTION_DOMAIN_POINTS, DATA_FOLDER
from src.utils import get_project_root, verify_path_exists

# Set the root directory for the project
root = get_project_root()


def create_variables(data):
    """
    Create input (x) and output (y) variables from the dataset.

    Args:
        data (np.ndarray): The dataset containing input and output values.

    Returns:
        tuple: Two numpy arrays, x (inputs) and y (outputs).
    """
    print("creating variables...")
    repetitions = int(len(data) / FUNCTION_DOMAIN_POINTS)
    x = list()
    y = list()

    for k in range(repetitions):
        x.append(data[k * FUNCTION_DOMAIN_POINTS, :4])
        y.append(
            data[k * FUNCTION_DOMAIN_POINTS : (k + 1) * FUNCTION_DOMAIN_POINTS, 5:9]
        )

    x = np.array(x)
    y = np.array(y)

    return x, y


def split(x, y):
    """
    Split the dataset into training and testing sets.

    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Output data.

    Returns:
        tuple: Four numpy arrays, x_train, x_test, y_train, and y_test.
    """
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


def get_fPCA_scores(profile, y_train, y_test, grid_points, evaluate=False):
    """
    Compute the Functional Principal Component Analysis (FPCA) scores.

    Args:
        profile (str): The profile to be evaluated (e.g., 'I', 'Q', 'U', 'V').
        y_train (np.ndarray): Training output data.
        y_test (np.ndarray): Testing output data.
        grid_points (np.ndarray): Grid points for the functional data.
        evaluate (bool, optional): If True, evaluate the reconstructions. Defaults to False.

    Returns:
        tuple: Train scores, test scores, and the FPCA object.
    """
    n_components = hparams.FPCA.N_COMPONENTS
    train_matrix = separate_response(y_train, profile)
    test_matrix = separate_response(y_test, profile)

    print(f"computing fPCA for profile {profile}...")
    fdata_train = skfda.FDataGrid(data_matrix=train_matrix, grid_points=grid_points)
    fdata_test = skfda.FDataGrid(data_matrix=test_matrix, grid_points=grid_points)

    fpca = FPCA(n_components=n_components)
    train_scores = fpca.fit_transform(fdata_train)
    test_scores = fpca.transform(fdata_test)

    if evaluate:
        evaluate_reconstructions(fpca, fdata_test, test_scores, profile)

    return train_scores, test_scores, fpca


def evaluate_reconstructions(fpca, fdata, scores, profile):
    """
    Evaluate the reconstructions from the FPCA.

    Args:
        fpca (FPCA): The FPCA object.
        fdata (FDataGrid): Functional data grid.
        scores (np.ndarray): Scores from FPCA transformation.
        profile (str): The profile being evaluated (e.g., 'I', 'Q', 'U', 'V').

    Returns:
        None
    """
    print("evaluating fPCA reconstructions for profile", profile)
    reconstructions = fpca.inverse_transform(scores)
    MSE = mean_squared_error(fdata, reconstructions)
    print(f"Mean Squared Error for {profile}:", MSE)


def separate_response(data, profile):
    """
    Separate the response data for a given profile.

    Args:
        data (np.ndarray): The response data.
        profile (str): The profile to be separated (e.g., 'I', 'Q', 'U', 'V').

    Returns:
        np.ndarray: The separated response data for the given profile.
    """
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
    """
    Main function to process the dataset, compute FPCA scores, and save the results.
    """
    # Load the data
    print("loading data...")
    data = np.loadtxt(os.path.join(root, DATASET_PATH), delimiter=",")
    grid_points = data[:FUNCTION_DOMAIN_POINTS, 4].reshape(1, -1)

    # Create variables from the data
    x, y = create_variables(data)

    # Normalize the input data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    joblib.dump(scaler, os.path.join(root, MODEL_SAVE_FOLDER, "scaler.pkl"))

    # Split the data into training and testing sets
    x_train, x_test, y_obs_train, y_obs_test = split(x, y)

    # Save the variables
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
    # Compute and save FPCA scores for each profile
    for profile in ["I", "Q", "U", "V"]:
        train_scores, test_scores, fpca = get_fPCA_scores(
            profile, y_obs_train, y_obs_test, grid_points
        )
        prof_obs_test = separate_response(y_obs_test, profile)

        print(f"saving {profile} scores...")
        save_folder = os.path.join(root, DATA_FOLDER, profile)
        verify_path_exists(save_folder)
        np.savetxt(
            os.path.join(save_folder, f"y_train.csv"), train_scores, delimiter=","
        )
        np.savetxt(os.path.join(save_folder, f"y_test.csv"), test_scores, delimiter=",")
        np.savetxt(
            os.path.join(save_folder, f"y_obs_test.csv"), prof_obs_test, delimiter=","
        )
        model_folder = os.path.join(root, MODEL_SAVE_FOLDER, profile)
        joblib.dump(fpca, os.path.join(model_folder, f"fpca.pkl"))


if __name__ == "__main__":
    main()
