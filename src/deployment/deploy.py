import os

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from skfda.preprocessing.dim_reduction import FPCA

from configs.NN_config import hparams, SAVE_FOLDER
from configs.data import DATA_FOLDER
from src.models.functional_neural_networks.MLP import MLP
from src.utils import get_project_root

# Set the root directory for the project
root = get_project_root()


def load_fpca():
    """
    Load the Functional Principal Component Analysis (FPCA) models for all profiles.

    Returns:
        list: A list of FPCA objects for profiles 'I', 'Q', 'U', and 'V'.
    """
    fpcas = []
    for profile in ["I", "Q", "U", "V"]:
        fpca = joblib.load(os.path.join(root, DATA_FOLDER, profile, "fpca.pkl"))
        fpcas.append(fpca)
    return fpcas


def load_models():
    """
    Load the pre-trained MLP models for all profiles.

    Returns:
        list: A list of MLP model objects for profiles 'I', 'Q', 'U', and 'V'.
    """
    models = []
    for profile in ["I", "Q", "U", "V"]:
        model_state_dict = torch.load(
            os.path.join(root, SAVE_FOLDER, profile, hparams.DEPLOYMENT.MODEL_NAME)
        )
        model = MLP()
        model.load_state_dict(model_state_dict)
        models.append(model)
    return models


def forward_pass(scaler, fpcas, models, inputs):
    """
    Perform a forward pass through the models to get predictions.

    Args:
        scaler (object): Scaler object used to normalize the inputs.
        fpcas (list): List of FPCA objects for each profile.
        models (list): List of MLP models for each profile.
        inputs (np.ndarray): Array of input parameters.

    Returns:
        list: A list of predicted outputs for each profile.
    """
    # Transform inputs using the scaler
    inputs = scaler.transform(inputs.reshape(1, -1))
    prediction = []
    for model, fpca in zip(models, fpcas):
        model.eval()
        input_array = torch.from_numpy(inputs).to(torch.float32)
        with torch.no_grad():
            scores = model(input_array)
            scores = scores.numpy()
            preds = fpca.inverse_transform(scores)
            prediction.append(preds)
    return prediction


def main():
    """
    Main function to load models, perform predictions, and plot the results.
    """
    # Load the scaler
    scaler = joblib.load(os.path.join(root, DATA_FOLDER, "scaler.pkl"))

    # Load FPCAs and models
    fpcas = load_fpca()
    models = load_models()

    # Input parameters for the prediction
    T_eff = 3600
    B = 500
    gamma = 85
    xi = 33
    inputs = np.array([T_eff, B, gamma, xi])

    # Perform the forward pass to get predictions
    output = forward_pass(scaler, fpcas, models, inputs)

    # Plot predictions for each profile
    for profile, pred in zip(["I", "Q", "U", "V"], output):
        fig, ax = plt.subplots()
        pred.plot(axes=ax, label="prediction", color="red", linestyle="dashed")

        plt.title(
            f"Prediction for {profile}, T_eff={T_eff}, B={B}, gamma={gamma}, xi={xi}"
        )
        plt.ylim(-1, 1)
        ax.legend()
        plt.show()


if __name__ == "__main__":
    main()
