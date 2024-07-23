import gpytorch
import torch

from configs.GP_config import hparams
from src.models.gaussian_process.GP_params import prior_mean, kernel


class GPWithNNFeatureExtractor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPWithNNFeatureExtractor, self).__init__(train_x, train_y, likelihood)
        self.mean_module = prior_mean
        self.covar_module = kernel

        input_dim = hparams.ENCODER.INPUT_DIM
        hidden_dim = hparams.ENCODER.HIDDEN_DIM
        output_dim = hparams.ENCODER.LATENT_DIM

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim),
        )

    def separate_concatenate(self, x):
        configs = x[:, :4]
        configs = self.feature_extractor(configs)
        return torch.cat((configs, x[:, 4:]), dim=1)

    def forward(self, x):
        x = self.separate_concatenate(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if hparams.MODEL.MULTITASK:
            distribution = (
                gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                    gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                )
            )
        else:
            distribution = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return distribution
