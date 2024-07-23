import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from configs.data import DATASET_PATH

data = np.loadtxt(DATASET_PATH, delimiter=",")
print("read data: ", data.shape)

x = data[:, :5]


configs = x[:, :4]
scaler = StandardScaler()
configs = scaler.fit_transform(configs)
pca = PCA(n_components=1)
configs_ = pca.fit_transform(configs)

x = np.concatenate([configs_, data[:, 4].reshape(-1, 1)], axis=1).astype(np.float16)
print("transformed inputs: ", x.shape)
I = data[:, 5].astype(np.float16)

V = data[:, 6]
Q = data[:, 7]
U = data[:, 8]

x_train, x_test, I_train, I_test = train_test_split(x, I, test_size=0.2, random_state=42)


GP = GaussianProcessRegressor(
    n_restarts_optimizer=0,
    random_state=42,
    normalize_y=True,
    kernel=RBF() + WhiteKernel(),
    copy_X_train=False,
)
print("training...")
print(x_train.dtype, I_train.dtype)
GP.fit(x_train, I_train)

print("testing...")
score = GP.score(x_test, I_test)
print(score)

