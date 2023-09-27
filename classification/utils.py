import numpy as np


def load_data_train(oversample=True):
    AGN_data = np.load('../infoVAE/test_results/latent/AGN.npy')
    AGN_labels = np.full(AGN_data.shape[0], 'AGN')

    NOAGN_data = np.load('../infoVAE/test_results/latent/NOAGN.npy')
    NOAGN_labels = np.full(NOAGN_data.shape[0], 'NOAGN') 

    if oversample:
        UHD_data = np.load('../infoVAE/test_results/latent/UHD_2times.npy') # load oversampled UHD
    else:
        UHD_data = np.load('../infoVAE/test_results/latent/UHD.npy')
    UHD_labels = np.full(UHD_data.shape[0], 'UHD') 

    if oversample:
        n80_data = np.load('../infoVAE/test_results/latent/n80_2times.npy') # load oversampled n80
    else:
        n80_data = np.load('../infoVAE/test_results/latent/n80.npy')
    n80_labels = np.full(n80_data.shape[0], 'n80')

    mockobs_0915_data = np.load('../infoVAE/test_results/latent/mockobs_0915.npy')
    mockobs_0915_labels = np.full(mockobs_0915_data.shape[0], 'mockobs_0915')

    X = np.concatenate((AGN_data, NOAGN_data, UHD_data, n80_data, mockobs_0915_data), axis=0)
    y = np.concatenate((AGN_labels, NOAGN_labels, UHD_labels, n80_labels, mockobs_0915_labels), axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]

    return X, y