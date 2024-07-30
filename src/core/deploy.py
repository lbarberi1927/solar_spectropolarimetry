import os

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from skfda.preprocessing.dim_reduction import FPCA

from configs.NN_config import hparams, MODEL_SAVE_FOLDER
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
        fpca = joblib.load(os.path.join(root, MODEL_SAVE_FOLDER, profile, "fpca.pkl"))
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
            os.path.join(root, MODEL_SAVE_FOLDER, profile, hparams.DEPLOYMENT.MODEL_NAME)
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


def plot_predictions(output, input_data):
    for profile, pred in zip(["I", "Q", "U", "V"], output):
        fig, ax = plt.subplots()
        pred.plot(axes=ax, label="prediction", color="red")

        plt.title(
            f"Prediction for {profile}, input {input_data}"
        )
        plt.ylim(-1, 1)
        plt.show()


def validate_input(input_data, input_ranges, expected_num_inputs=4):
    if len(input_data) != expected_num_inputs:
        raise ValueError(f"Expected {expected_num_inputs} inputs, but got {len(input_data)} inputs.")

    for i, value in enumerate(input_data):
        if not (input_ranges[i][0] <= value <= input_ranges[i][1]):
            raise ValueError(
                f"Input value {value} at index {i} is out of range. Expected range for input {i} is {input_ranges[i]}.")


def main():
    """
    Main function to load models, perform predictions, and plot the results.
    """
    # Load the scaler
    scaler = joblib.load(os.path.join(root, MODEL_SAVE_FOLDER, "scaler.pkl"))

    # Load FPCAs and models
    fpcas = load_fpca()
    models = load_models()

    input_ranges = [(3600, 6000), (0, 5000), (0, 180), (0, 180)]
    expected_num_inputs = 4
    print(f"Please enter {expected_num_inputs} inputs with the following ranges:")
    print_ranges = {0: "Effective temperature (T_eff)", 1: "Magnetic field strength (B)",
                    2: "Gamma angle (gamma)", 3: "Xi angle (xi)"}
    for i, r in enumerate(input_ranges):
        print(f"{print_ranges[i]}: {r}")

    while True:
        try:
            user_input = input("Enter input data (comma-separated): T_eff, B, gamma, xi\n")
            if user_input.lower() == 'exit':
                print("Exiting the prediction loop.")
                break

            # Convert the user input to the appropriate format
            input_data = list(map(float, user_input.split(',')))
            validate_input(input_data, input_ranges, expected_num_inputs)

            # Make prediction
            prediction = forward_pass(scaler, fpcas, models, np.array(input_data))
            plot_predictions(prediction, input_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue


if __name__ == "__main__":
    main()
