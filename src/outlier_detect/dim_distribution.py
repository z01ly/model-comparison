import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import os
import time

from PIL import Image

import src.outlier_detect.utils as utils


def plot_all_lines(savepath_prefix, nz, model_str_list, sdss_test_df_path, plt_tuple):
    sampled_sdss_test_data = utils.sample_sdss_for_plot(sdss_test_df_path, nz, frac=0.3)

    mock_data_inlier_dict = {}
    mock_data_outlier_dict = {}
    for model_str in model_str_list:
        load_dir = os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep')
        mock_data_inlier_dict[model_str] = utils.stack_mock_train_test(load_dir, model_str, nz, 'inlier')
        mock_data_outlier_dict[model_str] = utils.stack_mock_train_test(load_dir, model_str, nz, 'outlier')

    fig, axes = plt.subplots(plt_tuple[0], plt_tuple[1], figsize=plt_tuple[2])
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c', 'm', 'orange']

    for i in range(nz):
        sns.kdeplot(sampled_sdss_test_data[:, i], ax=axes[i], color='black', label='sdss')

        for pos, model_str in enumerate(model_str_list):
            sns.kdeplot(mock_data_inlier_dict[model_str][:, i], ax=axes[i], color=colors[pos], label=model_str + '-inlier')
            sns.kdeplot(mock_data_outlier_dict[model_str][:, i], ax=axes[i], color=colors[pos], linestyle='dotted', label=model_str + '-outlier')
        
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-1.25, 1.25])
        axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath_prefix, 'outlier-detect', 'distribution-plot', 'plot-all-lines.png'))
    plt.close()


def plot_sdss_hist(savepath_prefix, nz, model_str_list, sdss_test_df_path, plt_tuple):
    sampled_sdss_test_data = utils.sample_sdss_for_plot(sdss_test_df_path, nz, frac=0.3)
    print(sampled_sdss_test_data.shape)

    fig, axes = plt.subplots(plt_tuple[0], plt_tuple[1], figsize=plt_tuple[2])
    axes = axes.flatten()

    for i in range(nz):
        axes[i].hist(sampled_sdss_test_data[:, i], bins=200, histtype='step', linewidth=2, alpha=0.6, color='g')
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-5, 5])
        # axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath_prefix, 'outlier-detect', 'distribution-plot', 'plot-sdss-hist.png'))
    plt.close()


def corner_plot(savepath_prefix, nz, model_str, sdss_test_df_path, frac):
    sdss_test_df = pd.read_pickle(sdss_test_df_path)
    sdss_test_df_sampled = sdss_test_df.sample(frac=0.025, random_state=42)
    sdss_test_df_sampled['label'] = 'sdss'
    # print(sdss_test_df_sampled.head())
    
    load_dir = os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep')

    train_inlier_df = pd.read_pickle(os.path.join(load_dir, 'train', 'inlier', model_str + '.pkl'))
    train_inlier_df['label'] = 'inlier'
    train_inlier_df.pop('mahalanobis')
    train_inlier_df_sampled = train_inlier_df.sample(frac=frac, random_state=42)
    # print(train_inlier_df.head(1))

    # train_outlier_df = pd.read_pickle(os.path.join(load_dir, 'train', 'outlier', model_str + '.pkl'))
    # train_outlier_df['label'] = 'outlier'
    # train_outlier_df.pop('mahalanobis')
    # train_outlier_df_sampled = train_outlier_df.sample(frac=frac, random_state=42)
    # print(train_outlier_df.head(1))

    # df = pd.concat([sdss_test_df_sampled, train_inlier_df_sampled, train_outlier_df_sampled], axis=0)
    
    
    df = pd.concat([sdss_test_df_sampled, train_inlier_df_sampled], axis=0)
    df.reset_index(drop=True, inplace=True)
    print(f"{model_str} shape: {df.shape}")

    custom_palette = {'sdss': 'orange', 'inlier': 'blue', 'outlier': 'skyblue'}
    g = sns.pairplot(df, hue='label', corner=True, palette=custom_palette) # vars: every column with a numeric datatype
    g.map_lower(sns.kdeplot, levels=3, color=".2")
    plt.savefig(os.path.join(savepath_prefix, 'outlier-detect', 'corner-plot', model_str + '.png'))
    plt.close()
