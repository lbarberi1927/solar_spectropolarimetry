import os
import gpytorch
import numpy as np
import torch

from config import hparams, DATA_FOLDER
from src.models.gaussian_process_raw import ExactGPModel
from src.models.gaussian_process_with_encoder import GPWithNNFeatureExtractor
from src.utils import get_project_root
from src.models.GP_params import likelihood

root = get_project_root()

def training_epoch(train_x, train_y):
    model = GPWithNNFeatureExtractor(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    training_iter = hparams.TRAIN.ITER
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.OPTIMIZER.LR)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(training_iter):
        with torch.autocast(device_type=device, dtype=torch.float16):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

    if hparams.TRAIN.SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(root, "logs", hparams.TRAIN.NAME))


def main():
    x_train_path = os.path.join(root, DATA_FOLDER, "x_train.csv")
    y_train_path = os.path.join(root, DATA_FOLDER, "y_train.csv")

    if hparams.TRAIN.FAST_RUN:
        x_train = np.loadtxt(x_train_path, delimiter=",", max_rows=hparams.TRAIN.ROWS)
        y_train = np.loadtxt(y_train_path, delimiter=",", max_rows=hparams.TRAIN.ROWS)
    else:
        x_train = np.loadtxt(x_train_path, delimiter=",")
        y_train = np.loadtxt(y_train_path, delimiter=",")

    x_train = torch.tensor(x_train).to(torch.float32)
    if hparams.MODEL.MULTITASK:
        y_train = torch.tensor(y_train).to(torch.float32)
    else:
        y_train = torch.tensor(y_train[:, 0]).to(torch.float32)

    np.random.seed(hparams.SEED)
    torch.manual_seed(hparams.SEED)

    training_epoch(x_train, y_train)



if __name__ == '__main__':
    main()
