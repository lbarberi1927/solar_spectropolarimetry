import torch
import gpytorch
from torch import nn

from configs.GP_config import hparams
from src.models.gaussian_process.GP_params import prior_mean, kernel


# Define the neural network encoder
class ConfigEncoder(nn.Module):
    def __init__(self):
        super(ConfigEncoder, self).__init__()
        input_dim = hparams.ENCODER.INPUT_DIM + 1 if hparams.ENCODE_DELTA_LAMBDA else hparams.ENCODER.INPUT_DIM
        self.fc1 = nn.Linear(input_dim, hparams.ENCODER.HIDDEN_DIM)
        self.fc2 = nn.Linear(hparams.ENCODER.HIDDEN_DIM, hparams.ENCODER.LATENT_DIM)

    def forward_nn(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def separate_concatenate(self, x):
        configs = x[:, :4]
        configs = self.forward_nn(configs)
        return torch.cat((configs, x[:, 4:]), dim=1)

    def forward(self, x):
        if hparams.ENCODE_DELTA_LAMBDA:
            return self.forward_nn(x)
        else:
            return self.separate_concatenate(x)


# Define the SVGP model
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        if hparams.MODEL.MULTITASK:
            variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
                inducing_points.size(-2),
                batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM]),
            )
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                ),
                num_tasks=hparams.MODEL.OUTPUT_DIM,
            )
        else:
            variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True
            )
        super().__init__(variational_strategy)
        self.mean_module = prior_mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

