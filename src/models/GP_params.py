import gpytorch
import torch

from config import hparams

unitask_prior_mean = {
    "CONSTANT": gpytorch.means.ConstantMean(),
    "ZERO": gpytorch.means.ZeroMean(),
}

unitask_kernel = {
    "RBF": gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    "MATERN": gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()),
}

multitask_prior_mean = {
    "CONSTANT": gpytorch.means.ConstantMean(batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM])),
    "ZERO": gpytorch.means.ZeroMean(batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM])),
}

multitask_kernel = {
    "RBF": gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM])),
        batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM]),
    ),
    "MATERN": gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM])),
        batch_shape=torch.Size([hparams.MODEL.OUTPUT_DIM]),
    ),
}



if hparams.MODEL.MULTITASK:
    prior_mean = multitask_prior_mean[hparams.MODEL.PRIOR_MEAN]
    kernel = multitask_kernel[hparams.MODEL.KERNEL]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=hparams.MODEL.OUTPUT_DIM)
else:
    prior_mean = unitask_prior_mean[hparams.MODEL.PRIOR_MEAN]
    kernel = unitask_kernel[hparams.MODEL.KERNEL]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
