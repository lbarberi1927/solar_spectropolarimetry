import os
import numpy as np
from sklearn.model_selection import train_test_split
from configs.GP_config import hparams
from src.utils import get_project_root, verify_path_exists
from sklearn.preprocessing import StandardScaler

from configs.data import DATASET_PATH, DATA_FOLDER

root = get_project_root()
data_path = os.path.join(root, DATASET_PATH)
data = np.loadtxt(data_path, delimiter=",")
print("read data: ", data.shape)

x = data[:, :5].astype(np.float16)
y = data[:, 5:].astype(np.float16)

if hparams.PREPROCESS.NORMALIZE == True:
    print("normalizing data...")
    scaler = StandardScaler()
    x[:, :4] = scaler.fit_transform(x[:, :4])
    y = scaler.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=hparams.TRAIN.TEST_SIZE,
    random_state=hparams.SEED,
)
print(
    "separated data: training - ",
    x_train.shape,
    y_train.shape,
    "testing - ",
    x_test.shape,
    y_test.shape,
)

print("saving data...")
save_path = os.path.join(root, DATA_FOLDER)
verify_path_exists(save_path)
np.savetxt(os.path.join(root, DATA_FOLDER, "x_train.csv"), x_train, delimiter=",")
np.savetxt(os.path.join(root, DATA_FOLDER, "x_test.csv"), x_test, delimiter=",")
np.savetxt(os.path.join(root, DATA_FOLDER, "y_train.csv"), y_train, delimiter=",")
np.savetxt(os.path.join(root, DATA_FOLDER, "y_test.csv"), y_test, delimiter=",")
