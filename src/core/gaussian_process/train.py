import os
from pathlib import Path

import gpytorch
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from configs.GP_config import hparams
from configs.data import DATA_FOLDER
from src.models.gaussian_process.gaussian_process_with_encoder import GPWithNNFeatureExtractor
from src.models.gaussian_process.variational_GP import SVGPModel, ConfigEncoder
from src.utils import get_project_root, verify_path_exists
from src.models.gaussian_process.GP_params import likelihood

import gc

gc.collect()
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

root = get_project_root()
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_standard_model(x_train, y_train):
    model = GPWithNNFeatureExtractor(x_train, y_train, likelihood)
    if hparams.TRAIN.PRE_TRAINED:
        state_dict = torch.load(os.path.join(root, "logs", hparams.TRAIN.EXISTING_NAME))
        model.load_state_dict(state_dict)

    model.train()
    likelihood.train()

    training_iter = hparams.TRAIN.ITER
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.OPTIMIZER.LR)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(
        torch.device(device)
    )
    model = model.to(torch.device(device))
    x_train = x_train.to(torch.device(device))
    y_train = y_train.to(torch.device(device))

    for i in range(training_iter):
        with torch.autocast(device_type=device, dtype=torch.float16):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
            optimizer.step()

    if hparams.TRAIN.SAVE_MODEL:
        save_path = os.path.join(root, "logs", hparams.TRAIN.NAME)
        verify_path_exists(Path(save_path).parent)
        torch.save(model.state_dict(), save_path)


def train_variational_model(x_train, y_train, inducing_points):
    encoder = ConfigEncoder()
    inducing_points = encoder(inducing_points)

    model = SVGPModel(inducing_points)
    if hparams.TRAIN.PRE_TRAINED:
        model_state_dict = torch.load(
            os.path.join(root, "logs", hparams.TRAIN.EXISTING_NAME)
        )
        model.load_state_dict(model_state_dict)

        encoder_state_dict = torch.load(
            os.path.join(
                root,
                "logs",
                hparams.TRAIN.EXISTING_NAME.replace(".pth", "_encoder.pth"),
            )
        )
        encoder.load_state_dict(encoder_state_dict)

    model = model.to(torch.device(device))

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=hparams.VARIATIONAL.BATCH_SIZE)

    if hparams.OPTIMIZER.TYPE == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
                {"params": encoder.parameters()},
            ],
            lr=hparams.OPTIMIZER.LR,
        )
    elif hparams.OPTIMIZER.TYPE == "L-BFGS":
        params = (
            list(model.parameters())
            + list(likelihood.parameters())
            + list(encoder.parameters())
        )
        optimizer = torch.optim.LBFGS(params, lr=hparams.OPTIMIZER.LR)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0)).to(
        torch.device(device)
    )

    model.train()
    likelihood.train()
    print(f"Before training: {torch.cuda.memory_allocated()}, {torch.cuda.mem_get_info()}", flush=True)

    for epoch in range(hparams.VARIATIONAL.EPOCHS):
        n_batches = 0
        acc_loss = 0
        for batch_x, batch_y in iter(dataloader):
            batch_x = encoder(batch_x)
            batch_x = batch_x.to(torch.device(device))
            batch_y = batch_y.to(torch.device(device))
            print(f"Batch: {n_batches}, {torch.cuda.memory_allocated()}", flush=True)
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                output = model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()
                acc_loss += loss.item()
                n_batches += 1
        print(
            "Epoch %d/%d - Mean Loss: %.3f"
            % (epoch + 1, hparams.VARIATIONAL.EPOCHS, acc_loss / n_batches),
            flush=True,
        )

    print(f"After training: {torch.cuda.memory_allocated()}", flush=True)
    if hparams.TRAIN.SAVE_MODEL:
        torch.save(model.state_dict(), os.path.join(root, "logs", hparams.TRAIN.NAME))
        torch.save(
            encoder.state_dict(),
            os.path.join(
                root, "logs", hparams.TRAIN.NAME.replace(".pth", "_encoder.pth")
            ),
        )


def load_data():
    x_train_path = os.path.join(root, DATA_FOLDER, "x_train.csv")
    y_train_path = os.path.join(root, DATA_FOLDER, "y_train.csv")

    if hparams.TRAIN.FAST_RUN:
        x_train = np.loadtxt(x_train_path, delimiter=",", max_rows=hparams.TRAIN.ROWS)
        y_train = np.loadtxt(y_train_path, delimiter=",", max_rows=hparams.TRAIN.ROWS)
    else:
        x_train = np.loadtxt(x_train_path, delimiter=",")
        y_train = np.loadtxt(y_train_path, delimiter=",")

    x_train = torch.from_numpy(x_train).to(torch.float32)
    if hparams.MODEL.MULTITASK:
        y_train = torch.from_numpy(y_train).to(torch.float32)
    else:
        y_train = torch.from_numpy(y_train[:, 0]).to(torch.float32)

    inducing_points = np.loadtxt(
        os.path.join(root, DATA_FOLDER, "inducing_points.csv"), delimiter=",", max_rows=25000
    )
    inducing_points = torch.from_numpy(inducing_points).to(
        torch.float32
    )

    return x_train, y_train, inducing_points


def main():
    print(
        f"preparing training with params: \n"
        f"Multitask: {str(hparams.MODEL.MULTITASK)} \n"
        f"Kernel: {hparams.MODEL.KERNEL} \n"
        f"Learning Rate: {hparams.OPTIMIZER.LR} \n"
        f"Encoder Output Dimension: {hparams.ENCODER.LATENT_DIM} \n"
        f"Pre-Trained Model: {str(hparams.TRAIN.PRE_TRAINED)} \n"
        f"Saving model: {str(hparams.TRAIN.SAVE_MODEL)} to {hparams.TRAIN.NAME} \n"
        f"Variational Model Enabled: {str(hparams.VARIATIONAL.ENABLE)}\n"
        f"Device: {device} \n",
        flush=True,
    )

    print(f"In the beninging: {torch.cuda.memory_allocated(), torch.cuda.mem_get_info()}")

    x_train, y_train, inducing_points = load_data()

    np.random.seed(hparams.SEED)
    torch.manual_seed(hparams.SEED)

    if hparams.VARIATIONAL.ENABLE:
        train_variational_model(x_train, y_train, inducing_points)
    else:
        train_standard_model(x_train, y_train)


if __name__ == "__main__":
    main()
