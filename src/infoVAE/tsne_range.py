import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import src.infoVAE.utils as utils


def save_tsne_results(model_list=[], is_sdss_test=False):
    if is_sdss_test:
        latent_z = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
        z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)
        np.save('src/infoVAE/tsne/tsne_sdss_test', z_embedded)
    else:
        for model_str in model_list:
            latent_z = utils.stack_train_val(model_str)
            z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)
            np.save('src/infoVAE/tsne/tsne_' + model_str, z_embedded)



def plot_tsne(key, model_list, include_sdss=False):
    filename_embedded_list = ['tsne_' + model_str + '.npy' for model_str in model_list]

    if include_sdss:
        filename_embedded_list.append('tsne_sdss_test.npy')

    fig = plt.figure(figsize=(10,10))

    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    markers = ['o', 'v', 's', '*', '+', 'h']
    i = 0
    for filename_embedded in filename_embedded_list:
        z_embedded = np.load('src/infoVAE/tsne/' + filename_embedded)
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])
        i += 1

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f"{key}: Applying tsne to latent z")
    plt.legend()


    if include_sdss:
        fig.savefig('src/infoVAE/tsne/' + key + '_and_sdss.png')
    else:
        fig.savefig('src/infoVAE/tsne/' + key + '.png')

    

def plot_compare(compare_dict):
    fig = plt.figure(figsize=(10,10))
    c_list = ['b', 'y']
    marker_list = ['o', 'v']
    keys_list = list(compare_dict.keys())

    for itr, (label, model_list) in enumerate(compare_dict.items()):
        filename_embedded_list = ['tsne_' + model_str + '.npy' for model_str in model_list]
        load_to_list = [np.load('src/infoVAE/tsne/' + model) for model in filename_embedded_list]
        all_tsne = np.concatenate(load_to_list, axis=0)

        plt.scatter(all_tsne[:, 0], all_tsne[:, 1], s=5, c=c_list[itr], marker=marker_list[itr], label=label)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f"Compare {keys_list[0]} and {keys_list[1]}")
    plt.legend()
    
    fig.savefig('src/infoVAE/tsne/compare_' + keys_list[0] + '_' + keys_list[1] + '.png')





if __name__ == "__main__":
    pass
    # model_list = ['TNG50-1', 'TNG100-1', 'illustris-1']
    # save_tsne_results(model_list)


