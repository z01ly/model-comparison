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



def plot_per_model(model_str):
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    mock_data = src.infoVAE.utils.stack_train_val(model_str)

    fig, axes = plt.subplots(4, 8, figsize=(32, 16))
    axes = axes.flatten()

    for i in range(sdss_test_data.shape[1]):
        sns.kdeplot(sdss_test_data[:, i], ax=axes[i], color='blue', label='sdss')
        sns.kdeplot(mock_data[:, i], ax=axes[i], color='orange', label=model_str.split('_')[0])
        axes[i].set_title(f'Dimension {i}')
        axes[i].set_xlim([-2, 2])
        axes[i].legend(fontsize= "x-small", loc="upper right")
        axes[i].set_xlabel('Values')
        axes[i].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(os.path.join('src/dim/distribution-plot/per-model', model_str.split('_')[0] + '.png'))
    plt.close()



def plot_per_dim(model_names):
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')

    mock_data_list = []
    for model_str in model_names:
        mock_data = src.infoVAE.utils.stack_train_val(model_str)
        mock_data_list.append(mock_data)

    for i in range(sdss_test_data.shape[1]):
        fig, axes = plt.subplots(2, int(len(model_names) / 2), figsize=(20, 20))
        axes = axes.flatten()

        for j, model_str in enumerate(model_names):
            sns.kdeplot(sdss_test_data[:, i], ax=axes[j], color='blue', label='sdss')
            sns.kdeplot(mock_data_list[j][:, i], ax=axes[j], color='orange', label=model_str.split('_')[0])
            axes[j].set_title(model_str.split('_')[0])
            axes[j].set_xlim([-5, 5])
            axes[j].legend(fontsize= "x-small", loc="upper right")
            axes[j].set_xlabel('Values')
            axes[j].set_ylabel('Density')

        plt.tight_layout()
        plt.savefig(os.path.join('src/dim/distribution-plot/per-dim', 'dim' + str(i) + '.png'))
        plt.close()



def dim_plot_concat(dim):
    dir_path_1 = 'src/dim/distribution-plot/per-dim'
    dir_path_2 = 'src/dim/dim-meaning/TNG100-1_snapnum_099/vector_0'
    output_dir = 'src/dim/distribution-plot/per-dim-concat'

    for i in range(dim):
        image1 = Image.open(os.path.join(dir_path_1, 'dim' + str(i) + '.png'))
        image2 = Image.open(os.path.join(dir_path_2, 'dim' + str(i) + '.png'))
        image2 = image2.resize((2880, 360))

        new_width = max(image1.width, image2.width)
        image1 = image1.resize((new_width, image1.height))

        new_height = image1.height + image2.height

        new_image = Image.new("RGB", (new_width, new_height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1.height))

        new_image.save(os.path.join(output_dir, 'dim' + str(i) + '.png'))




def corner_plot(model_str):
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    y_sdss_test = np.full((sdss_test_data.shape[0],), 'sdss_test')

    mock_data = src.infoVAE.utils.stack_train_val(model_str)
    y_mock = np.full(mock_data.shape[0], model_str.split('_')[0])
    
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
    model_names = ['AGNrt', 'NOAGNrt', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099', 'UHDrt', 'n80rt']
    # for model_str in model_names:
    #     plot_per_model(model_str)

    plot_per_dim(model_names)

    # dim_plot_concat(32)