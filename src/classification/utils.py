import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_data_df(model_str_list, load_data_dir, z_dim):
    X_list = []
    y_list = []
    for model_str in model_str_list:
        data_df = pd.read_pickle(os.path.join(load_data_dir, model_str + '.pkl'))
        
        data = data_df.iloc[:, 0:z_dim].to_numpy()
        X_list.append(data)
        y_list.append(np.full(data.shape[0], model_str))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]

    return X, y
