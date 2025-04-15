import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from src.deprecated.outlier_detect.mahalanobis import MDist
import src.deprecated.outlier_detect.dim_distribution as dim_distribution


def m_distance(savepath_prefix, nz, model_str_list):
    for key1 in ['train', 'test']:
        os.makedirs(os.path.join(savepath_prefix, 'outlier-detect', 'm-distance', key1), exist_ok=True)
        for key2 in ['inlier', 'outlier']:
            os.makedirs(os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', key1, key2), exist_ok=True)

    with open(os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'print-message.txt'), "w") as txt_file:
        txt_file.write(f"Percentage of inliers and outliers in each simulation model. \n")
        txt_file.write(f"\n\n")

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    for key in ['train', 'test']:
        mahal_dist = MDist(savepath_prefix, model_str_list, nz, sdss_test_df_path, key, alpha=0.95)
        mahal_dist()



def distribution(savepath_prefix, nz, model_str_list, plt_tuple):
    os.makedirs(os.path.join(savepath_prefix, 'outlier-detect', 'distribution-plot'), exist_ok=True)

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    dim_distribution.plot_all_lines(savepath_prefix, nz, model_str_list, sdss_test_df_path, plt_tuple)
    dim_distribution.plot_sdss_hist(savepath_prefix, nz, model_str_list, sdss_test_df_path, plt_tuple)



def distribution_corner(savepath_prefix, nz, model_str_dict):
    os.makedirs(os.path.join(savepath_prefix, 'outlier-detect', 'corner-plot'), exist_ok=True)

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')

    for model_str, frac in model_str_dict.items():
        dim_distribution.corner_plot(savepath_prefix, nz, model_str, sdss_test_df_path, frac)
        


if __name__ == '__main__':
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    # m_distance(savepath_prefix, nz, model_str_list)

    distribution(savepath_prefix, nz, model_str_list, (8, 4, (20, 40)))
    # distribution(savepath_prefix, nz, model_str_list, (1, 2, (12, 6)))
    # distribution(savepath_prefix, nz, model_str_list, (1, 3, (18, 6)))
    # distribution(savepath_prefix, nz, model_str_list, (2, 2, (12, 12)))

    # model_str_dict  = {'AGNrt': 0.8, 'NOAGNrt': 0.8, 'TNG100': 0.1, 'TNG50': 0.8, 'UHDrt': 1.0, 'n80rt': 1.0}
    # distribution_corner(savepath_prefix, nz, model_str_dict)
