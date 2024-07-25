from yacs.config import CfgNode as CN

SAVE_FOLDER = "logs"

hparams = CN()

hparams.SEED = 42

hparams.FPCA = CN()
hparams.FPCA.N_COMPONENTS = 14

hparams.MODEL = CN()
hparams.MODEL.INPUT_DIM = 4
hparams.MODEL.OUTPUT_DIM = hparams.FPCA.N_COMPONENTS
hparams.MODEL.HIDDEN_LAYERS = 2
hparams.MODEL.HIDDEN_DIM = 55

hparams.TRAIN = CN()
hparams.TRAIN.TEST_SIZE = 0.2
hparams.TRAIN.LOSS = "MSE"
hparams.TRAIN.EPOCHS = 10
hparams.TRAIN.LEARNING_RATE = 1e-3
hparams.TRAIN.BATCH_SIZE = 64
hparams.TRAIN.PRE_TRAINED = False
hparams.TRAIN.SAVE_MODEL = True
hparams.TRAIN.NAME = "dl_pm1p5/4_55_55_14_nn_10_epochs.pth"
hparams.TRAIN.EXISTING_NAME = "basic_nn_5_epochs.pth"

hparams.EVAL = CN()
hparams.EVAL.BATCH_SIZE = 32
hparams.EVAL.NAME = hparams.TRAIN.NAME
hparams.EVAL.PLOT = True
hparams.EVAL.EXTENDED_EVAL = True
