import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import os
import time

from PIL import Image

import src.infoVAE.utils


def plot(model_str_list, sdss_test_data_path, z_dim, key):
    sdss_test_data = np.load(sdss_test_data_path)
    mock_data_dict = {}
    for model_str in model_str_list:
        mock_data_dict[model_str] = src.infoVAE.utils.stack_mock_train_test('src/results/latent-vectors', model_str, z_dim, key)

    fig, axes = plt.subplots(4, 8, figsize=(64, 32))
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c', 'm', 'orange']
    
    for i in range(z_dim):
        sns.kdeplot(sdss_test_data[:, i], ax=axes[i], color='black', label='sdss')
        for pos, model_str in enumerate(model_str_list):
            sns.kdeplot(mock_data_dict[model_str][:, i], ax=axes[i], color=colors[pos], label=model_str)
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-2, 2])
        axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join('src/results/dim/distribution-plot', 'plot-' + key + '.png'))
    plt.close()


def plot_all_lines(model_str_list, sdss_test_data_path, z_dim):
    sdss_test_data = np.load(sdss_test_data_path)
    mock_data_inlier_dict = {}
    mock_data_outlier_dict = {}
    for model_str in model_str_list:
        mock_data_inlier_dict[model_str] = src.infoVAE.utils.stack_mock_train_test('src/results/latent-vectors', model_str, z_dim, 'inlier')
        mock_data_outlier_dict[model_str] = src.infoVAE.utils.stack_mock_train_test('src/results/latent-vectors', model_str, z_dim, 'outlier')

    fig, axes = plt.subplots(4, 8, figsize=(64, 32))
    axes = axes.flatten()
    colors = ['b', 'g', 'r', 'c', 'm', 'orange']

    for i in range(z_dim):
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
    plt.savefig(os.path.join('src/results/dim/distribution-plot', 'plot_all_lines.png'))
    plt.close()



def plot_inlier_outlier_compare(model_str_list, sdss_test_data_path, z_dim):
    sdss_test_data = np.load(sdss_test_data_path)
    mock_data_inlier_list = []
    mock_data_outlier_list = []
    for model_str in model_str_list:
        mock_data_inlier_list.append(src.infoVAE.utils.stack_mock_train_test('src/results/latent-vectors', model_str, z_dim, 'inlier'))
        mock_data_outlier_list.append(src.infoVAE.utils.stack_mock_train_test('src/results/latent-vectors', model_str, z_dim, 'outlier'))

    mock_data_inlier = np.concatenate(mock_data_inlier_list, axis=0)
    mock_data_outlier = np.concatenate(mock_data_outlier_list, axis=0)

    fig, axes = plt.subplots(4, 8, figsize=(64, 32))
    axes = axes.flatten()

    for i in range(z_dim):
        sns.kdeplot(sdss_test_data[:, i], ax=axes[i], color='black', label='sdss')
        sns.kdeplot(mock_data_inlier[:, i], ax=axes[i], color='skyblue', label='inlier')
        sns.kdeplot(mock_data_outlier[:, i], ax=axes[i], color='y', label='outlier')
        
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-2, 2])
        axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join('src/results/dim/distribution-plot', 'plot-inlier-outlier-compare.png'))
    plt.close()


def plot_per_dim(model_str_list, sdss_test_data_path, z_dim):
    sdss_test_data = np.load(sdss_test_data_path)

    mock_data_dict = {}
    mock_data_inlier_dict = {}
    for model_str in model_str_list:
        mock_data_dict[model_str] = src.infoVAE.utils.stack_mock_train_test(model_str, z_dim, 'all')
        mock_data_inlier_dict[model_str] = src.infoVAE.utils.stack_mock_train_test(model_str, z_dim, 'inlier')

    for i in range(z_dim):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes = axes.flatten()
        colors = ['b', 'g', 'r', 'c', 'm', 'orange']

        sns.kdeplot(sdss_test_data[:, i], ax=axes[0], color='black', label='sdss')
        for pos, model_str in enumerate(model_str_list):
            sns.kdeplot(mock_data_dict[model_str][:, i], ax=axes[0], color=colors[pos], label=model_str)
        axes[0].set_title('All')
        axes[0].set_xlim([-2, 2])
        axes[0].legend(loc="upper right")
        axes[0].set_xlabel('Values')
        axes[0].set_ylabel('Density')

        sns.kdeplot(sdss_test_data[:, i], ax=axes[1], color='black', label='sdss')
        for pos, model_str in enumerate(model_str_list):
            sns.kdeplot(mock_data_inlier_dict[model_str][:, i], ax=axes[1], color=colors[pos], label=model_str)
        axes[1].set_title('Inlier')
        axes[1].set_xlim([-2, 2])
        axes[1].legend(loc="upper right")
        axes[1].set_xlabel('Values')
        axes[1].set_ylabel('Density')

        plt.tight_layout()
        plt.savefig(os.path.join('src/results/dim/distribution-plot/per-dim', 'dim' + str(i) + '.png'))
        plt.close()



def plot_per_dim_concat(dim):
    dir_path_1 = 'src/results/dim/distribution-plot/per-dim'
    dir_path_2 = 'src/dim/dim-meaning/TNG100-1_snapnum_099/vector_0'
    output_dir = 'src/results/dim/distribution-plot/per-dim-concat'

    for i in range(dim):
        image1 = Image.open(os.path.join(dir_path_1, 'dim' + str(i) + '.png'))
        image2 = Image.open(os.path.join(dir_path_2, 'dim' + str(i) + '.png'))
        image2 = image2.resize((2400, 300))

        new_width = max(image1.width, image2.width)
        image1 = image1.resize((new_width, image1.height))

        new_height = image1.height + image2.height

        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))

        new_image.save(os.path.join(output_dir, 'dim' + str(i) + '.png'))




def corner_plot(model_str, sdss_test_data_path, z_dim):
    sdss_test_data = np.load(sdss_test_data_path)
    y_sdss_test = np.full((sdss_test_data.shape[0],), 'sdss_test')

    mock_data = src.infoVAE.utils.stack_mock_train_test(model_str, z_dim, only_inlier=False)
    y_mock = np.full(mock_data.shape[0], model_str)
    
    data_combined = np.vstack((mock_data, sdss_test_data))
    y_combined = np.concatenate((y_mock, y_sdss_test))

    df = pd.DataFrame(data_combined, columns=[f'f_{i}' for i in range(32)])
    df['y'] = y_combined
    
    # plt.figure(figsize=(12, 6))
    start_time = time.time()
    sns.pairplot(df, vars=df.columns[: -1], hue='y')
    end_time = time.time()
    
    # plt.title(f"Corner plot of {model_str.split('_')[0]}")
    plt.savefig(os.path.join('src/dim/corner-plot/', model_str.split('_')[0] + '.png'))
    plt.close()

    print("Execution time: ", end_time - start_time, "seconds")
    


if __name__ == '__main__':
    # model_names = ['AGNrt', 'NOAGNrt', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099', 'UHDrt', 'n80rt']

    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    sdss_test_data_path='src/results/latent-vectors/sdss_test.npy'
    # for key in ['all', 'inlier', 'outlier']:
    #     plot(model_str_list, sdss_test_data_path, 32, key)
    plot_all_lines(model_str_list, sdss_test_data_path, 32)
    # plot_inlier_outlier_compare(model_str_list, sdss_test_data_path, 32)
    # plot_per_dim(model_str_list, sdss_test_data_path, 32)
    # plot_per_dim_concat(32)