import os
from yacs.config import CfgNode as CN


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = 'data.nosync'
DATASET_PATH = os.path.join(DATA_FOLDER, 'fe6302_basic.csv')

hparams = CN()

hparams.SEED = 42

hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 1e-5

hparams.MODEL = CN()
hparams.MODEL.KERNEL = "RBF + MATERN"
hparams.MODEL.PRIOR_MEAN = "CONSTANT"
# hparams.MODEL.LIKELIHOOD = "GAUSSIAN"
hparams.MODEL.MULTITASK = True
hparams.MODEL.OUTPUT_DIM = 4

hparams.ENCODER = CN()
hparams.ENCODER.INPUT_DIM = 4
hparams.ENCODER.HIDDEN_DIM = 2
hparams.ENCODER.LATENT_DIM = 1

hparams.TRAIN = CN()
hparams.TRAIN.ITER = 5
hparams.TRAIN.FAST_RUN = False
hparams.TRAIN.ROWS = 1000
hparams.TRAIN.TEST_SIZE = 0.2
hparams.TRAIN.SAVE_MODEL = True
hparams.TRAIN.NAME = "basic/cluster_run.pth"
hparams.TRAIN.PRE_TRAINED = False
hparams.TRAIN.EXISTING_NAME = "basic_configs/attempt_zero_5_iter.pth"

hparams.VARIATIONAL = CN()
hparams.VARIATIONAL.ENABLE = True
hparams.VARIATIONAL.N_INDUCING_POINTS = 10000
hparams.VARIATIONAL.BATCH_SIZE = 64
hparams.VARIATIONAL.EPOCHS = 5
hparams.VARIATIONAL.K_MEANS_BATCH = 100000
hparams.VARIATIONAL.INDUCING_ALGORITHM = "minibatch_kmeans"

hparams.LOSS = CN()
hparams.LOSS.TYPE = "mll"

hparams.PREPROCESS = CN()
hparams.PREPROCESS.NORMALIZE = True

hparams.TEST = CN()
hparams.TEST.ROWS = 100
