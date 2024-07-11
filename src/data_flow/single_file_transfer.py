import os
import numpy as np

new_path = os.path.join(os.getcwd(), '../../../sveta/syntspec/fe6302/')
os.chdir(new_path)
print(os.getcwd())

file_path = "fe6302_lte"
feless_path = "_lte"
g_conf = ["4000", "4100", "4200", "4300", "4400", "4500", "4600", "4700", "4800", "4900", "5000",
          "5100", "5200", "5400", "5500", "5600", "5700", "5800", "5900", "6000"]
#+ g45v1_
G_conf = ["0000", "0500", "1000", "1500", "2000", "2500", "3000", "3500", "4000", "4500", "5000"]
# G_g
g_2_conf = ["000", "030", "060", "090", "120", "150", "180"]
# _c
c_conf = ["000", "022", "045", "067", "090", "112", "135", "157", "180"]

all_data = list()

for g in g_conf:
    for G in G_conf:
        for g_2 in g_2_conf:
            for c in c_conf:
                file = file_path + g + "g45v1_" + G + "G_g" + g_2 + "_c" + c + ".dat"
                if not os.path.exists(file):
                    file = feless_path + g + "g45v1_" + G + "G_g" + g_2 + "_c" + c + ".dat"
                try:
                    data = np.loadtxt(file, skiprows=2)
                except:
                    print("File not found: ", file)

                configs = np.array([[int(g), int(G), int(g_2), int(c)]])
                rep_configs = np.tile(configs, (data.shape[0], 1))

                new_data = np.concatenate([rep_configs, data], axis=1)
                all_data.append(new_data)


all_data = np.vstack(all_data)
print(all_data.shape)

print("saving to barberi")
sink = os.path.join(os.getcwd(), '../../../barberi/data/fe6302_basic.csv')
np.savetxt(sink, all_data, delimiter=",")
