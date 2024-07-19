import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

data = np.loadtxt('../data.nosync/fe6302_lte4400g45v1_4000G_g150_c022.dat', skiprows=2)

n_samples = 100
indices = np.random.choice(data.shape[0], size=n_samples, replace=False)
data = data[indices]

x = data[:, 0]
I = data[:, 1]
V = data[:, 2]
Q = data[:, 3]
U = data[:, 4]

configs = np.array([[4400, 4000, 150, 22]])
rep_configs = np.tile(configs, (len(x), 1))
input = np.concatenate([x.reshape(-1,1), rep_configs], axis=1)

GP = GaussianProcessRegressor(n_restarts_optimizer=5, random_state=42, normalize_y=True, kernel=RBF() + WhiteKernel())
GP.fit(input, I)

# Step 4: Make predictions
X_pred = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
X_pred = np.concatenate([X_pred, rep_configs[0].reshape(1, -1).repeat(1000, axis=0)], axis=1)
y_pred, sigma = GP.predict(X_pred, return_std=True)

# Step 5: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x, I, 'r.', markersize=10, label='Training data')
plt.plot(X_pred[:, 0], y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred[:, 0].ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma,
                 alpha=0.2, color='grey', label='95% confidence interval')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()
