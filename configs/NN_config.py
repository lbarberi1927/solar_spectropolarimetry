from yacs.config import CfgNode as CN

# Define the folder to save logs
SAVE_FOLDER = "logs"

# Initialize hyperparameters configuration node
hparams = CN()

# Set random seed for reproducibility
hparams.SEED = 42

# FPCA (Functional Principal Component Analysis) configuration
hparams.FPCA = CN()
hparams.FPCA.N_COMPONENTS = 14  # Number of components for FPCA

# Model configuration
hparams.MODEL = CN()
hparams.MODEL.INPUT_DIM = 4  # Input dimension for the model
hparams.MODEL.OUTPUT_DIM = hparams.FPCA.N_COMPONENTS  # Output dimension (based on FPCA components)
hparams.MODEL.HIDDEN_LAYERS = 2  # Number of hidden layers
hparams.MODEL.HIDDEN_DIM = 55  # Dimension of hidden layers

# Training configuration
hparams.TRAIN = CN()
hparams.TRAIN.TEST_SIZE = 0.2  # Proportion of test data
hparams.TRAIN.LOSS = "MSE"  # Loss function
hparams.TRAIN.EPOCHS = 10  # Number of training epochs
hparams.TRAIN.LEARNING_RATE = 1e-3  # Learning rate
hparams.TRAIN.BATCH_SIZE = 64  # Batch size for training
hparams.TRAIN.PRE_TRAINED = False  # Flag indicating if a pre-trained model should be used
hparams.TRAIN.SAVE_MODEL = True  # Flag to save the model after training
hparams.TRAIN.NAME = "increased_temp_range/4_55_55_14_nn_10_epochs.pth"  # Name of the saved model
hparams.TRAIN.EXISTING_NAME = "basic_nn_5_epochs.pth"  # Name of the existing pre-trained model

# Evaluation configuration
hparams.EVAL = CN()
hparams.EVAL.BATCH_SIZE = 32  # Batch size for evaluation
hparams.EVAL.NAME = hparams.TRAIN.NAME  # Name of the model to be evaluated
hparams.EVAL.PLOT = True  # Flag to plot evaluation results
hparams.EVAL.EXTENDED_EVAL = True  # Flag for extended evaluation

# Deployment configuration
hparams.DEPLOYMENT = CN()
hparams.DEPLOYMENT.MODEL_NAME = hparams.EVAL.NAME  # Name of the model to be deployed
