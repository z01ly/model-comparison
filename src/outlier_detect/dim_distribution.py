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
    sdss_test_df = pd.read_pickle(sdss_test_df_path)
    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()

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
        sns.kdeplot(sdss_test_data[:, i], ax=axes[i], color='black', label='sdss')
        for pos, model_str in enumerate(model_str_list):
            sns.kdeplot(mock_data_inlier_dict[model_str][:, i], ax=axes[i], color=colors[pos], label=model_str + '-inlier')
            sns.kdeplot(mock_data_outlier_dict[model_str][:, i], ax=axes[i], color=colors[pos], linestyle='dotted', label=model_str + '-outlier')
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-2, 2])
        axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath_prefix, 'outlier-detect', 'distribution-plot', 'plot-all-lines.png'))
    plt.close()
