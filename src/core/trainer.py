import os
import gpytorch
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader

from config import hparams, DATA_FOLDER
from src.models.gaussian_process_raw import ExactGPModel
from src.models.gaussian_process_with_encoder import GPWithNNFeatureExtractor
from src.models.variational_GP import ConfigEncoder, SVGPModel
from src.utils import get_project_root
from src.models.GP_params import likelihood

root = get_project_root()

def train_standard_model(x_train, y_train):
    model = GPWithNNFeatureExtractor(x_train, y_train, likelihood)
    if hparams.TRAIN.PRE_TRAINED:
        state_dict = torch.load(os.path.join(root, "logs", hparams.TRAIN.EXISTING_NAME))
        model.load_state_dict(state_dict)

    model.train()
    likelihood.train()

    training_iter = hparams.TRAIN.ITER
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.OPTIMIZER.LR)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(training_iter):
        with torch.autocast(device_type=device, dtype=torch.float16):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

    if hparams.TRAIN.SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(root, "logs", hparams.TRAIN.NAME))


def train_variational_model(x_train, y_train):

    kmeans = KMeans(n_clusters=hparams.VARIATIONAL.N_INDUCING_POINTS)
    cluster_points = np.array(x_train)
    kmeans.fit(cluster_points)
    inducing_points = torch.tensor(kmeans.cluster_centers_).to(torch.float32)

    model = SVGPModel(inducing_points)
    dataset = TensorDataset(torch.tensor(x_train), *y_train)
    dataloader = DataLoader(dataset, batch_size=hparams.VARIATIONAL.BATCH_SIZE)

    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=hparams.OPTIMIZER.LR)
    print(model.state_dict())
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

    model.train()
    likelihood.train()

    for epoch in hparams.VARIATIONAL.EPOCHS:
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            output = model(batch_x)
            loss = -mll(output, batch_y)
            loss.backward()
            optimizer.step()

    if hparams.TRAIN.SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(root, "logs", hparams.TRAIN.NAME))

def load_data():
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

    return x_train, y_train

def main():
    print(f"preparing training with params: \n "
          f"Multitask: {str(hparams.MODEL.MULTITASK)} \n"
          f"Kernel: {hparams.MODEL.KERNEL} \n"
          f"Learning Rate: {hparams.OPTIMIZER.LR} \n"
          f"Encoder Output Dimension: {hparams.ENCODER.LATENT_DIM} \n"
          f"Pre-Trained Model: {str(hparams.TRAIN.PRE_TRAINED)} \n"
          f"Saving model: {str(hparams.TRAIN.SAVE_MODEL)} to {hparams.TRAIN.NAME} \n"
          f"Variational Model Enabled: {str(hparams.VARIATIONAL.ENABLE)}")
    x_train, y_train = load_data()

    np.random.seed(hparams.SEED)
    torch.manual_seed(hparams.SEED)

    if hparams.VARIATIONAL.ENABLE:
        train_variational_model(x_train, y_train)
    else:
        train_standard_model(x_train, y_train)



if __name__ == '__main__':
    main()
