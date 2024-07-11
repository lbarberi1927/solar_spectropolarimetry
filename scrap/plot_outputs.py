import numpy as np
import matplotlib.pyplot as plt

def extract_data(file_path, param):
    data = np.loadtxt(file_path, skiprows=2)
    if param == 'I':
        return data[:, :2]
    elif param == 'V':
        return data[:, [0, 2]]
    elif param == 'Q':
        return data[:, [0, 3]]
    elif param == 'U':
        return data[:, [0, 4]]
    return data

for param in ['I', 'V', 'Q', 'U']:
    data = extract_data('../data/fe6302_lte4400g45v1_4500G_g000_c022.dat', param)
    data_ = extract_data('../data/fe6302_lte4400g45v1_4000G_g150_c022.dat', param)
    data__ = extract_data('../data/fe6302_lte6000g45v1_3500G_g180_c112.dat', param)

    plt.title(param)
    plt.plot(data[:, 0], data[:, 1], color='red', label='4400g, 4500G, g000, c022')
    plt.plot(data_[:, 0], data_[:, 1], color='blue', label='4400g, 4000G, g150, c022')
    plt.plot(data__[:, 0], data__[:, 1], color='green', label='6000g, 3500G, g180, c112')
    plt.legend()
    plt.show()

