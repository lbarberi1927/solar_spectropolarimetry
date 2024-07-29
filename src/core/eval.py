import os

import joblib
import numpy as np
import skfda
import torch
from skfda.misc.scoring import mean_squared_error
from skfda.preprocessing.dim_reduction import FPCA
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from configs.NN_config import hparams, SAVE_FOLDER
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

# Set the root directory for the project
root = get_project_root()


def load_labels(profile, obs="", extension="csv"):
    """
    Load labels for a given profile from a CSV file.

    Args:
        profile (str): Profile name (e.g., 'I', 'Q', 'U', 'V').
        obs (str, optional): Observation type, defaults to empty string.
        extension (str, optional): File extension, defaults to 'csv'.

    Returns:
        torch.Tensor: Loaded labels as a torch tensor.
    """
    y = np.loadtxt(
        os.path.join(root, DATA_FOLDER, profile, f"y_{obs}test.{extension}"),
        delimiter=",",
    )
    y = torch.from_numpy(y).to(torch.float32)
    return y


def evaluate(model, test_loader, obs_loader, fpca):
    """
    Evaluate the model on test data.

    Args:
        model (torch.nn.Module): Trained MLP model.
        test_loader (DataLoader): DataLoader for the test data.
        obs_loader (DataLoader): DataLoader for the observation data.
        fpca (FPCA): Functional Principal Component Analysis object.

    Returns:
        None
    """
    model.eval()
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        MSE_loss = 0.0
        f_MSE_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            MSE_loss += loss.item()

        avg_MSE_loss = MSE_loss / len(test_loader)
        print(f"Test MSE Loss: {avg_MSE_loss:.8f}")

        if hparams.EVAL.EXTENDED_EVAL:
            for inputs, y_obs in obs_loader:
                outputs = model(inputs)
                rec_outputs = fpca.inverse_transform(outputs.numpy())

                original_outputs = skfda.FDataGrid(
                    data_matrix=y_obs, grid_points=rec_outputs.grid_points
                )
                f_MSE_loss += mean_squared_error(rec_outputs, original_outputs)

        avg_fMSE_loss = f_MSE_loss / len(test_loader)
        print(f"Test L2 Loss: {avg_fMSE_loss:.8f}")


def plot(model, x, y, y_obs, profile, index, fpca):
    """
    Plot the model predictions against the actual observations and reconstructions.

    Args:
        model (torch.nn.Module): Trained MLP model.
        x (torch.Tensor): Input data tensor.
        y (torch.Tensor): Target data tensor.
        y_obs (torch.Tensor): Observation data tensor.
        profile (str): Profile name (e.g., 'I', 'Q', 'U', 'V').
        index (int): Index of the sample to plot.
        fpca (FPCA): Functional Principal Component Analysis object.

    Returns:
        None
    """
    x = x[index].unsqueeze(0)
    y = y[index].numpy()
    y_obs = y_obs[index].numpy()

    y = fpca.inverse_transform(y)
    y_obs = skfda.FDataGrid(data_matrix=y_obs, grid_points=y.grid_points)

    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy()
        y_pred = fpca.inverse_transform(y_pred)

        # Plot both FDataGrid objects
        fig, ax = plt.subplots()
        y_obs.plot(axes=ax, color="black", label="Observation")
        y.plot(axes=ax, color="blue", linestyle="dashed", label="Reconstruction")
        y_pred.plot(axes=ax, color="red", linestyle="dashed", label="Prediction")

        ax.set_title(f"Sample Observation, Reconstruction, and Prediction: {profile}")
        ax.legend()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Δλ")
        ax.set_ylabel(profile)
        plt.show()


def main():
    """
    Main function to prepare and evaluate the model on test data for multiple profiles.
    """
    index = 8677  # np.random.randint(0, 10000)
    # Load input test data
    x = np.loadtxt(os.path.join(root, DATA_FOLDER, "x_test.csv"), delimiter=",")
    x = torch.from_numpy(x).to(torch.float32)

    # Evaluate the model for each profile
    for profile in ["I", "Q", "U", "V"]:
        print(f"Evaluating profile {profile}")
        y = load_labels(profile)

        test_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=hparams.EVAL.BATCH_SIZE,
            shuffle=False,
        )

        # Load the pre-trained model
        model_state_dict = torch.load(
            os.path.join(root, SAVE_FOLDER, profile, hparams.EVAL.NAME)
        )
        model = MLP()
        model.load_state_dict(model_state_dict)

        if hparams.EVAL.EXTENDED_EVAL:
            y_obs = load_labels(profile, "obs_")

            # Load the FPCA model
            fpca = joblib.load(os.path.join(root, DATA_FOLDER, profile, "fpca.pkl"))

            # Plot the predictions
            plot(model, x, y, y_obs, profile, index, fpca)
            obs_loader = DataLoader(
                TensorDataset(x, y_obs),
                batch_size=hparams.EVAL.BATCH_SIZE,
                shuffle=False,
            )
        else:
            fpca = None
            obs_loader = None

        # Evaluate the model
        evaluate(model, test_loader, obs_loader, fpca)


if __name__ == "__main__":
    main()
