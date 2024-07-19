from yacs.config import CfgNode as CN

DATA_FOLDER = 'data.nosync'

hparams = CN()

hparams.SEED = 42

hparams.FPCA = CN()
hparams.FPCA.N_COMPONENTS = 14

hparams.MODEL = CN()
hparams.MODEL.INPUT_DIM = 4
hparams.MODEL.OUTPUT_DIM = hparams.FPCA.N_COMPONENTS
hparams.MODEL.HIDDEN_LAYERS = 2
hparams.MODEL.HIDDEN_DIM = 10

hparams.TRAIN = CN()
hparams.TRAIN.TEST_SIZE = 0.2
hparams.TRAIN.LOSS = "MSE"
hparams.TRAIN.EPOCHS = 10
hparams.TRAIN.LEARNING_RATE = 1e-3
hparams.TRAIN.BATCH_SIZE = 32
