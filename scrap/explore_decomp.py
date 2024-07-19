import os

import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt
from src.models.gaussian_process.gaussian_process_raw import ExactGPModel
from src.models.gaussian_process.GP_params import likelihood

from src.utils import get_project_root
from configs.GP_config import DATA_FOLDER, hparams

root = get_project_root()
device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.loadtxt(os.path.join(root, DATA_FOLDER, 'fe6302_lte4400g45v1_4500G_g000_c022.dat'), skiprows=2)
x = torch.from_numpy(data[:, 0])
y = torch.from_numpy(data[:, 1])

model = ExactGPModel(x, y, likelihood)

model.train()
likelihood.train()

training_iter = hparams.TRAIN.ITER
optimizer = torch.optim.Adam(model.parameters(), lr=hparams.OPTIMIZER.LR)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    with torch.autocast(device_type=device, dtype=torch.float16):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iter, loss.item()))
        optimizer.step()

print("GP params", model.state_dict())

new_model = ExactGPModel(x, y, likelihood)
new_model.load_state_dict(model.state_dict())
new_model.eval()

test_x = torch.linspace(-1.5, 1.5, 51)
with torch.no_grad():
    preds = new_model(test_x)
    mean = preds.mean
    sigma = preds.stddev

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'r.', markersize=10, label='Training data')
    plt.plot(test_x, mean, 'b-', label='Prediction')
    plt.fill_between(test_x.ravel(), mean - 1.96 * sigma, mean + 1.96 * sigma,
                     alpha=0.2, color='grey', label='95% confidence interval')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    plt.show()
