import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm, chi2

from src.infoVAE.mmdVAE_train import compute_mmd


class SNTest():
    def __init__(self, savepath_prefix, nz):
        self.savepath_prefix = savepath_prefix
        self.nz = nz
        self.sdss_test_df_path = os.path.join(self.savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
        self.sdss_test_df = pd.read_pickle(self.sdss_test_df_path)
        self.sdss_test_data = self.sdss_test_df.iloc[:, 0:self.nz].to_numpy()


    def plot(self, loc=0, scale=1):
        os.makedirs(os.path.join(self.savepath_prefix, 'sn-test'), exist_ok=True)
        print(os.path.join(self.savepath_prefix, 'sn-test'))

        fig, axs = plt.subplots(8, 4, figsize=(20, 40))
        axs = axs.ravel()

        for i in range(self.sdss_test_data.shape[1]):
            sns.histplot(self.sdss_test_data[:, i], kde=True, ax=axs[i], stat='density') #, common_norm=False)
            x = np.linspace(-5, 5, 100)
            axs[i].plot(x, norm.pdf(x, loc, scale), 'r-', lw=2)
            axs[i].set_xlim([-10, 10])
            axs[i].set_title(f'Dimension {i+1}')

        plt.tight_layout()
        fig.savefig(os.path.join(self.savepath_prefix, 'sn-test', f'sdss_norm_loc{loc}_scale{scale}.png'), bbox_inches='tight')


    def squared_m(self, alpha=0.95):
        threshold = chi2.ppf(alpha, self.nz)
        with open(os.path.join(self.savepath_prefix, 'sn-test', 'squared_m.txt'), 'w') as f:
            f.write(f'alpha: {alpha}, reference percentage: {1-alpha}, threshold: {threshold}\n')

        squared_m_dist = np.sum(self.sdss_test_data**2, axis=1)
        outlier_num = np.sum(squared_m_dist > threshold)
        with open(os.path.join(self.savepath_prefix, 'sn-test', 'squared_m.txt'), 'a') as f:
            f.write(f'The number of outliers: {outlier_num}\n')
            f.write(f'Outlier ratio: {outlier_num / len(squared_m_dist)}')


    def mmd(self, itr=50):
        sample_num = 400
        mmd_list = []
        for i in range(itr):
            sampled_idx = np.random.choice(self.sdss_test_data.shape[0], sample_num, replace=False)
            z = self.sdss_test_data[sampled_idx]
            z_tensor = torch.tensor(z, dtype=torch.float32)
            true_samples = torch.randn(sample_num, self.nz)
            mmd = compute_mmd(true_samples, z_tensor)
            mmd_list.append(mmd.item())

        mmd_mean = np.mean(mmd_list)
        with open(os.path.join(self.savepath_prefix, 'sn-test', 'mmd.txt'), 'w') as f:
            f.write(f'Average mmd loss (50 iterations): {mmd_mean}')


def mmd_test(std, itr=50):
    sample_num = 400
    mmd_list = []
    for i in range(itr):
        true_samples = torch.randn(sample_num, 1)
        samples = torch.normal(0, std, (sample_num, 1))
        mmd = compute_mmd(true_samples, samples)
        mmd_list.append(mmd.item())

    mmd_mean = np.mean(mmd_list)
    print("Average mmd", mmd_mean)



if __name__ == '__main__':
    nz = 32
    # savepath_prefix = 'results/' + str(nz) + '-dims'
    # savepath_prefix_2 = 'others/nll_0.02mmd/' + str(nz) + '-dims'
    # savepath_prefix_3 = 'others/0.1nll_1mmd/' + str(nz) + '-dims'

    # sn_test = SNTest(savepath_prefix_3, nz)
    # sn_test.mmd()

    mmd_test(0.1)
    mmd_test(0.5)
    mmd_test(0.9)
