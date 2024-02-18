import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from scipy.stats import chi2
from sklearn.covariance import MinCovDet



class MDist():
    def __init__(self, model_str, z_dim, data_df, sdss_test_data, distance_path, alpha=0.95):
        self.model_str = model_str
        self.alpha = alpha

        self.data_df = data_df
        self.data = data_df.iloc[:, 0:z_dim].to_numpy()
        self.sdss_test_data = sdss_test_data 

        self.z_dim = z_dim
        # #DoF of the chi-squared distribution equals #variables
        self.cutoff = chi2.ppf(self.alpha, self.z_dim)

        self.distance_path = distance_path


    def __call__(self):
        if os.path.exists(self.distance_path):
            distances = np.load(self.distance_path)
        else:
            print("Compute Mahalanobis distance...")
            distances = self.mahalanobis()
        
        all_indices = np.arange(self.data.shape[0])
        outlier_indices = np.where(distances > self.cutoff)[0]
        inlier_indices = np.setdiff1d(all_indices, outlier_indices)
        print(f"inlier ratio of {self.model_str}: {inlier_indices.shape[0] / all_indices.shape[0]}")

        inlier_df = self.data_df.iloc[inlier_indices]
        inlier_df.reset_index(drop=True, inplace=True)
        inlier_df.to_pickle(os.path.join('src/results/latent-vectors/train-inlier', self.model_str + '.pkl'))

        outlier_df = self.data_df.iloc[outlier_indices]
        outlier_df.reset_index(drop=True, inplace=True)
        outlier_df.to_pickle(os.path.join('src/results/latent-vectors/train-outlier', self.model_str + '.pkl'))


    def mahalanobis(self):
        # center = np.mean(self.sdss_test_data, axis=0)
        robust_cov = MinCovDet(random_state=0).fit(self.sdss_test_data)
        mahal_robust_cov = robust_cov.mahalanobis(self.data)
        np.save(self.distance_path, mahal_robust_cov)

        return mahal_robust_cov

    
    def print_cutoff(self):
        print(f"Cutoff point of alpha {self.alpha}: {self.cutoff}")
    



if __name__ == '__main__':
    model_names = ['AGNrt'] # , 'NOAGNrt', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099', 'UHDrt', 'n80rt']
    for model_str in model_names:
        mahal = MDist(model_str, 32, 
            pd.read_pickle('src/results/latent-vectors/train/' + model_str + '.pkl'),
            np.load('src/results/latent-vectors/sdss_test.npy'),
            os.path.join('src/outlier_detect/mahalanobis/save-distance', model_str + '.npy'))
        mahal()
