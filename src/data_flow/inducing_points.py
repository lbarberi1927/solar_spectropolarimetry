import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from config import hparams, DATA_FOLDER
from src.utils import get_project_root

root = get_project_root()


def get_inducing_points(x_train):
    kmeans = MiniBatchKMeans(
        n_clusters=hparams.VARIATIONAL.N_INDUCING_POINTS,
        verbose=3,
        batch_size=hparams.VARIATIONAL.K_MEANS_BATCH,
        random_state=hparams.SEED
    )
    kmeans.fit(x_train)
    print("found inducing points", flush=True)

    return kmeans.cluster_centers_


def main():
    x_train = np.loadtxt(os.path.join(root, DATA_FOLDER, "x_train.csv"), delimiter=",")
    inducing_points = get_inducing_points(x_train)

    np.savetxt(os.path.join(root, DATA_FOLDER, "inducing_points.csv"), inducing_points, delimiter=",")


if __name__ == '__main__':
    print("running inducing points script...")
    main()
