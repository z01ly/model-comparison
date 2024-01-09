import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re

from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch.utils.data import Dataset


def load_data(model_list, switch='train'):
    if switch == 'test':
        switch = 'val'
        pattern = re.compile(r'_[0-9]+times')
        model_list = [re.sub(pattern, '', model_str) for model_str in model_list]
        print(model_list)
        
    X_list = []
    y_list = []
    for model_str in model_list:
        # data = np.load('src/infoVAE/test_results/latent/' + switch + 'set_' + model_str + '.npy')
        # full_path_prefix
        data = np.load('/export/home/lzhou/src/infoVAE/test_results/latent/' + switch + 'set_' + model_str + '.npy')
        X_list.append(data)
        y_list.append(np.full(data.shape[0], model_str.split('_')[0])) # make model name shorter

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]

    return X, y


def print_msg(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    for class_label, int_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"Class '{class_label}' is transformed to encoding integer: {int_label}")


def pre_nn_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    return X_tensor, y_tensor



def pre_makedirs(key):
    dir_list = ['calibration-curve', 'confusion-matrix', 'save-model', 'violin-plot']
    # model_list = ['illustris', 'nihao', 'compare']
    for directory in dir_list:
        os.makedirs(os.path.join('src/classification', directory, key), exist_ok=True)



# A pytorch dataset for SimpleNN
class TrainValDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class TestDataset(Dataset):
    def __init__(self, test_data):
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx]



def train_loss_plot(loss_list, xlabel_str, save_path):
    fig = plt.figure(figsize=(10,8))
    plt.title("Training loss per " + xlabel_str)
    plt.plot(loss_list, label='Training Loss')

    plt.xlabel(xlabel_str)
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
     



if __name__ == '__main__':
    # pre_makedirs()
    
    # default oversample
    # nihao_list = ['AGN', 'NOAGN', 'UHD_2times', 'mockobs_0915', 'n80_2times']
    # illustris_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'illustris-1_snapnum_135']
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = load_data(model_names, switch='train')
    print(X.shape)
    print_msg(y)
    
    