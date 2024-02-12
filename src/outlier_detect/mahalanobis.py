import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from scipy.stats import chi2
from sklearn.covariance import MinCovDet



class MahalanobisDist():
    def __init__(self, model_str):
        self.model_str = model_str
        self.data = np.load('src/infoVAE/test_results/latent/' + 'trainset_' + self.model_str + '.npy')
        self.sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
        self.dim = self.data.shape[1]

        self.save_dist_path = os.path.join('src/outlier_detect/mahalanobis/save-distance', self.model_str + '.npy')


    def __call__(self, alpha=0.95):
        cutoff = chi2.ppf(alpha, self.dim)
        print(f"cutoff point of alpha {alpha}: {cutoff}")

        if os.path.exists(self.save_dist_path):
            distances = np.load(self.save_dist_path)
        else:
            distances = self.mahalanobis()
        
        outlier_idx = np.where(distances > cutoff)[0]
        # print(f"outlier indices shape of {model_str}: {outlier_idx.shape} \n")
        print((self.data.shape[0] - outlier_idx.shape[0]) / self.data.shape[0])


    def mahalanobis(self):
        # center = np.mean(self.sdss_test_data, axis=0)
        robust_cov = MinCovDet(random_state=0).fit(self.sdss_test_data)
        mahal_robust_cov = robust_cov.mahalanobis(self.data)
        np.save(self.save_dist_path, mahal_robust_cov)

        return mahal_robust_cov

    



if __name__ == '__main__':
    model_names = ['AGNrt', 'NOAGNrt', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099', 'UHDrt', 'n80rt']
    for model_str in model_names:
        mahal = MahalanobisDist(model_str)
        mahal()
