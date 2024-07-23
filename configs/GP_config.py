import os
from yacs.config import CfgNode as CN


hparams = CN()

hparams.SEED = 42

hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 1e-2

hparams.MODEL = CN()
hparams.MODEL.KERNEL = "RBF"
hparams.MODEL.PRIOR_MEAN = "CONSTANT"
# hparams.MODEL.LIKELIHOOD = "GAUSSIAN"
hparams.MODEL.MULTITASK = False
hparams.MODEL.OUTPUT_DIM = 1 if not hparams.MODEL.MULTITASK else 4

hparams.ENCODE_DELTA_LAMBDA = False
hparams.ENCODER = CN()
hparams.ENCODER.INPUT_DIM = 4
hparams.ENCODER.HIDDEN_DIM = 2
hparams.ENCODER.LATENT_DIM = 1
hparams.ENCODER.MODEL = "NN"  # PCA OR NN

hparams.TRAIN = CN()
hparams.TRAIN.ITER = 100
hparams.TRAIN.FAST_RUN = False
hparams.TRAIN.ROWS = 1000
hparams.TRAIN.TEST_SIZE = 0.2
hparams.TRAIN.SAVE_MODEL = True
hparams.TRAIN.NAME = "variational/cluster_run.pth"
hparams.TRAIN.PRE_TRAINED = False
hparams.TRAIN.EXISTING_NAME = "basic_configs/attempt_zero_5_iter.pth"

hparams.VARIATIONAL = CN()
hparams.VARIATIONAL.ENABLE = True
hparams.VARIATIONAL.N_INDUCING_POINTS = 50000
hparams.VARIATIONAL.BATCH_SIZE = 32
hparams.VARIATIONAL.EPOCHS = 5
hparams.VARIATIONAL.K_MEANS_BATCH = 100000
hparams.VARIATIONAL.INDUCING_ALGORITHM = "minibatch_kmeans"

hparams.LOSS = CN()
hparams.LOSS.TYPE = "mll"

hparams.PREPROCESS = CN()
hparams.PREPROCESS.NORMALIZE = True

hparams.TEST = CN()
hparams.TEST.ROWS = 100
