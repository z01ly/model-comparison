import numpy as np
import os

from torch.utils.data import Dataset


def load_data_train(model_list):
    X_list = []
    y_list = []
    for model_str in model_list:
        data = np.load('src/infoVAE/test_results/latent/trainset_' + model_str + '.npy')
        X_list.append(data)
        y_list.append(np.full(data.shape[0], model_str.split('_')[0])) # make model name shorter

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]

    return X, y



def pre_makedirs(key):
    dir_list = ['calibration-curve', 'confusion-matrix', 'save-model', 'violin-plot']
    # model_list = ['illustris', 'nihao', 'compare']
    for directory in dir_list:
        os.makedirs(os.path.join('src/classification', directory, key), exist_ok=True)



# A pytorch dataset for SimpleNN
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
     



if __name__ == '__main__':
    pass
    # pre_makedirs()
    
    # default oversample
    # nihao_list = ['AGN', 'NOAGN', 'UHD_2times', 'mockobs_0915', 'n80_2times']
    # illustris_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'illustris-1_snapnum_135']
    # X, y = load_data_train(nihao_list)
    
    