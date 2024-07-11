import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import hparams
from src.models.GP_params import prior_mean, kernel


# Define the neural network encoder
class ConfigEncoder(nn.Module):
    def __init__(self):
        super(ConfigEncoder, self).__init__()
        self.fc1 = nn.Linear(hparams.ENCODER.INPUT_DIM, hparams.ENCODER.HIDDEN_DIM)
        self.fc2 = nn.Linear(hparams.ENCODER.HIDDEN_DIM, hparams.ENCODER.LATENT_DIM)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the SVGP model
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=hparams.MODEL.OUTPUT_DIM,
        )
        super(SVGPModel, self).__init__(variational_strategy)
        self.mean_module = prior_mean
        self.covar_module = kernel
        self.encoder = ConfigEncoder()

    def separate_concatenate(self, x):
        configs = x[:, :4]
        configs = self.encoder(configs)
        return torch.cat((configs, x[:, 4:]), dim=1)

    def forward(self, x):
        x = self.separate_concatenate(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

