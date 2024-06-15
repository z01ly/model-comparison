import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def lower_dim(savepath_prefix, z_dim):
    os.makedirs(os.path.join(savepath_prefix, 'pca'), exist_ok=True)

    sdss_test_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')) 
    sdss_test_data = sdss_test_df.iloc[:, 0:z_dim].to_numpy()
    sc = StandardScaler()
    sdss_test_data = sc.fit_transform(sdss_test_data)

    dim_list = []
    sum_explained_var_ratio_list = []
    for n_components in range(1, z_dim+1):
        dim_list.append(n_components)
        pca = PCA(n_components=n_components)
        pca.fit(sdss_test_data)
        explained_variance_ratio_ = pca.explained_variance_ratio_
        sum_explained_var_ratio_list.append(np.sum(explained_variance_ratio_))

    fig = plt.figure(figsize=(12, 6))
    plt.plot(dim_list, sum_explained_var_ratio_list)
    plt.xticks(range(1, z_dim+2))
    plt.yticks([i/10 for i in range(11)])
    plt.ylabel('Sum of Explained variance Ratio')
    plt.xlabel('Number of Principal components')
    plt.title('Explained Variance Ratio vs. Number of PCA Components')
    plt.grid(True)
    # plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(savepath_prefix, 'pca', 'explained_var_ratio.png'), bbox_inches='tight')
    



if __name__ == "__main__":
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    lower_dim(savepath_prefix, nz)