import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from configs.NN_config import hparams, DATA_FOLDER
from src.utils import get_project_root

root = get_project_root()

def split(x, profile):
    y = np.loadtxt(os.path.join(root, DATA_FOLDER, profile, f"{profile}_scores.csv"), delimiter=",")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=hparams.TRAIN.TEST_SIZE,
        random_state=hparams.SEED,
    )
    print("separated data: training - ", x_train.shape, y_train.shape, "testing - ", x_test.shape, y_test.shape)

    return x_train, x_test, y_train, y_test


def main():
    x = np.loadtxt(os.path.join(root, DATA_FOLDER, "variables.csv"), delimiter=",")
    print("normalizing data...")
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    for profile in ["I", "Q", "U", "V"]:
        x_train, x_test, y_train, y_test = split(x, profile)
        print(f"split {profile} data")
        print("saving data...")
        np.savetxt(os.path.join(root, DATA_FOLDER, profile, f"x_train.csv"), x_train, delimiter=",")
        np.savetxt(os.path.join(root, DATA_FOLDER, profile, f"x_test.csv"), x_test, delimiter=",")
        np.savetxt(os.path.join(root, DATA_FOLDER, profile, f"y_train.csv"), y_train, delimiter=",")
        np.savetxt(os.path.join(root, DATA_FOLDER, profile, f"y_test.csv"), y_test, delimiter=",")
        print(f"saved {profile} data")


if __name__ == '__main__':
    main()
