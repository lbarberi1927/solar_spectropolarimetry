import gpytorch
from config import hparams
from src.models.gaussian_process.GP_params import prior_mean, kernel

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = prior_mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if hparams.MODEL.MULTITASK:
            distribution = gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )
        else:
            distribution = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return distribution


