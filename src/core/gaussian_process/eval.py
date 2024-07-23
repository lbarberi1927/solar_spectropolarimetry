import os

import gpytorch
import numpy as np
import torch

from src.models.gaussian_process.gaussian_process_raw import ExactGPModel
from src.utils import get_project_root
from configs.GP_config import hparams
from configs.data import DATA_FOLDER

root = get_project_root()


def load_model(model_name):
    x_train_path = os.path.join(root, DATA_FOLDER, "x_train.csv")
    y_train_path = os.path.join(root, DATA_FOLDER, "y_train.csv")

    x_train = np.loadtxt(x_train_path, delimiter=",")
    y_train = np.loadtxt(y_train_path, delimiter=",")
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train[:, 0])

    state_dict = torch.load(os.path.join(root, "logs", model_name))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    print("model created", model.state_dict())

    model.load_state_dict(state_dict)
    print("model loaded", model.state_dict())

    return model, likelihood


def predict(model, likelihood, x):
    print("evaluating...")

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))

    return observed_pred


def main():
    model, likelihood = load_model(hparams.TRAIN.NAME)
    observed_preds = predict(model, likelihood, model.train_inputs[0])

    print(observed_preds.mean, observed_preds.variance, model.train_inputs[0])


if __name__ == "__main__":
    main()
