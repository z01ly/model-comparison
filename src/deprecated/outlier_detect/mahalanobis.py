# ref: https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html

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
    def __init__(self, savepath_prefix, model_str_list, z_dim, sdss_test_df_path, key, alpha=0.95):
        self.savepath_prefix = savepath_prefix
        self.model_str_list = model_str_list
        self.alpha = alpha

        self.sdss_test_df = pd.read_pickle(sdss_test_df_path)

        self.z_dim = z_dim
        self.threshold = chi2.ppf(self.alpha, self.z_dim) # DoF of chi-squared distribution = number of variables
        self.key = key # 'train' or 'test'


    def __call__(self):
        self.print_threshold()
        print("Compute Mahalanobis distance >>>")
        self.mahalanobis()

        for model_str in self.model_str_list:
            df_with_dist = pd.read_pickle(os.path.join(self.savepath_prefix, 'outlier-detect', 'm-distance', self.key, model_str + '.pkl'))
            distances = df_with_dist['mahalanobis'].to_numpy()
            
            all_indices = np.arange(distances.shape[0])
            outlier_indices = np.where(distances > self.threshold)[0]
            inlier_indices = np.setdiff1d(all_indices, outlier_indices)

            with open(os.path.join(self.savepath_prefix, 'outlier-detect', 'in-out-sep', 'print-message.txt'), "a") as txt_file:
                txt_file.write(f"{model_str} {self.key} set: \n")
                txt_file.write(f"Inlier ratio {inlier_indices.shape[0] / all_indices.shape[0]} \n")
                txt_file.write(f"Outlier ratio {outlier_indices.shape[0] / all_indices.shape[0]} \n")
                txt_file.write(f"\n")

            inlier_df = df_with_dist.iloc[inlier_indices].copy()
            inlier_df.reset_index(drop=True, inplace=True)
            inlier_df.to_pickle(os.path.join(self.savepath_prefix, 'outlier-detect', 'in-out-sep', self.key, 'inlier', model_str + '.pkl'))

            outlier_df = df_with_dist.iloc[outlier_indices].copy()
            outlier_df.reset_index(drop=True, inplace=True)
            outlier_df.to_pickle(os.path.join(self.savepath_prefix, 'outlier-detect', 'in-out-sep', self.key, 'outlier', model_str + '.pkl'))


    def mahalanobis(self):
        sdss_test_data = self.sdss_test_df.iloc[:, 0:self.z_dim].to_numpy()
        # center = np.mean(self.sdss_test_data, axis=0)
        robust_cov = MinCovDet(random_state=42).fit(sdss_test_data)

        for model_str in self.model_str_list:
            df = pd.read_pickle(os.path.join(self.savepath_prefix, 'latent-vectors', self.key, model_str + '.pkl'))
            data = df.iloc[:, 0:self.z_dim].to_numpy()
        
            mahal_robust_cov = robust_cov.mahalanobis(data)
            df['mahalanobis'] = mahal_robust_cov

            m_dist_save_path = os.path.join(self.savepath_prefix, 'outlier-detect', 'm-distance', self.key, model_str + '.pkl')
            df.to_pickle(m_dist_save_path)

    
    def print_threshold(self):
        with open(os.path.join(self.savepath_prefix, 'outlier-detect', 'in-out-sep', 'print-message.txt'), "a") as txt_file:
            txt_file.write(f"Chi-squared distribution threshold of alpha {self.alpha}: {self.threshold}. \n")
            txt_file.write(f"\n")



if __name__ == '__main__':
    pass 
    