import os

DATA_FOLDER = 'data.nosync'
DATASET_PATH = os.path.join(DATA_FOLDER, 'fe6302_basic.csv')  # path to the dataset containing physical parameters,
# delta lambda and response variables (Stokes profiles)
SIRIUS_DATA_ORIGIN_FOLDER = os.path.join("sveta", "syntspec", "fe6302")  # path to the folder containing the raw data
# files
SIRIUS_DATA_DESTINATION_PATH = os.path.join("solar_spectropolarimetry", DATASET_PATH)  # path to the destination file on sirius
FUNCTION_DOMAIN_POINTS = 301  # number of points in the domain of the functional data
