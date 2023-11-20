import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import utils


def save_tsne_results(model_list):
    for model_str in model_list:
        latent_z = utils.stack_train_val(model_str)
        z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)
        np.save('./tsne/tsne_' + model_str, z_embedded)



def plot_tsne(key, include_sdss=False): 
    # key = 'illustris' or 'nihao'
    if key == 'illustris':
        filename_embedded_list = ['tsne_illustris-1.npy', 'tsne_TNG100-1.npy', 'tsne_TNG50-1.npy']
    elif key == 'nihao':
        filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy', 'tsne_mockobs_0915.npy']

    if include_sdss:
        filename_embedded_list.append('tsne_sdss_test.npy')

    fig = plt.figure(figsize=(10,10))

    colors = ['b', 'r', 'g', 'y', 'm', 'c']
    markers = ['o', 'v', 's', '*', '+', 'h']
    i = 0
    for filename_embedded in filename_embedded_list:
        z_embedded = np.load('./tsne/' + filename_embedded)
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])
        i += 1

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Applying tsne to latent z")
    plt.legend()


    if include_sdss:
        fig.savefig('./tsne/' + key + '_and_sdss.png')
    else:
        fig.savefig('./tsne/' + key + '.png')

    

def plot_compare():
    illustris = ['tsne_illustris-1.npy', 'tsne_TNG100-1.npy', 'tsne_TNG50-1.npy']
    nihao = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy', 'tsne_mockobs_0915.npy']
    load_illustris = [np.load('./tsne/' + model) for model in illustris]
    load_nihao = [np.load('./tsne/' + model) for model in nihao]
    all_illustris = np.concatenate(load_illustris, axis=0)
    all_nihao = np.concatenate(load_nihao, axis=0)

    fig = plt.figure(figsize=(10,10))

    plt.scatter(all_illustris[:, 0], all_illustris[:, 1], s=5, c='y', marker='o', label='illustris')
    plt.scatter(all_nihao[:, 0], all_nihao[:, 1], s=5, c='b', marker='v', label='nihao')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Applying tsne to latent z")
    plt.legend()

    fig.savefig('./tsne/compare_illustris_nihao.png')



def find_cover_range(key):
    if key == 'illustris':
        filename_embedded_list = ['tsne_illustris-1.npy', 'tsne_TNG100-1.npy', 'tsne_TNG50-1.npy']
    elif key == 'nihao':
        filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy', 'tsne_mockobs_0915.npy']


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    colors = ['b', 'r', 'g', 'y', 'k']
    markers = ['o', 'v', 's', '*', '+']
    i = 0
    arrays = []
    for filename_embedded in filename_embedded_list:
        z_embedded = np.load('./tsne/' + filename_embedded)
        arrays.append(z_embedded)

        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])
        i += 1

    tsne_data = np.concatenate(arrays, axis=0)

    pca = PCA(n_components=2)
    pca.fit(tsne_data)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
    center = np.mean(tsne_data, axis=0)
    major_axis_length = np.sqrt(eigenvalues[0]) * 2 * 2
    minor_axis_length = np.sqrt(eigenvalues[1]) * 2 * 2

    ellipse = Ellipse(
        xy=center,
        width=major_axis_length,
        height=minor_axis_length,
        angle=np.degrees(angle),
        edgecolor=colors[i],
        facecolor='none'
    )
    ax.add_patch(ellipse)
    
    ax.set_aspect('equal', adjustable='box') # Set aspect ratio to equal
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Fitting an ellipse to tsne data")
    plt.legend()

    fig.savefig('./tsne/' + key + '_cover_range.png')

    return center, major_axis_length, minor_axis_length, angle




def main(key, center, major_axis_length, minor_axis_length, angle):
    sdss_latent = np.load('./test_results/latent/sdss_test.npy')
    sdss_filenames = np.load('./test_results/latent/sdss_test_filenames.npy')
    sdss_tsne = np.load('./tsne/tsne_sdss_test.npy')

    xc, yc = center
    x_new = (sdss_tsne[:, 0] - xc) * np.cos(angle) + (sdss_tsne[:, 1] - yc) * np.sin(angle)
    y_new = (sdss_tsne[:, 1] - yc) * np.cos(angle) - (sdss_tsne[:, 0] - xc) * np.sin(angle)

    a = major_axis_length / 2
    b = minor_axis_length / 2
    inside_ellipse_mask = (x_new**2 / (a**2) + y_new**2 / (b**2)) <= 1

    indices = np.where(inside_ellipse_mask)[0]

    selected_sdss_embedded = sdss_tsne[indices]
    selected_sdss_latent = sdss_latent[indices]
    selected_sdss_filenames = sdss_filenames[indices]

    np.save('./test_results/latent/' + key + '_selected_sdss_test.npy', selected_sdss_latent)
    np.savetxt('./test_results/latent_txt/' + key + '_selected_sdss_test.txt', selected_sdss_latent, delimiter=',', fmt='%s')
    np.savetxt('./tsne/' + key + '_selected_sdss_filenames.txt', selected_sdss_filenames, fmt="%s")


    if key == 'illustris':
        filename_embedded_list = ['tsne_illustris-1.npy', 'tsne_TNG100-1.npy', 'tsne_TNG50-1.npy']
    elif key == 'nihao':
        filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy', 'tsne_mockobs_0915.npy']

    fig = plt.figure(figsize=(10,10))
    colors = ['b', 'r', 'g', 'k']
    markers = ['o', 'v', 's', '*']
    i = 0
    for filename_embedded in filename_embedded_list:
        z_embedded = np.load('./tsne/' + filename_embedded)
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])
        i += 1

    plt.scatter(selected_sdss_embedded[:, 0], selected_sdss_embedded[:, 1], s=5, c='y', marker='+', label='selected sdss test')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Selected sdss test")
    plt.legend()

    fig.savefig('./tsne/' + key + '_select_sdss_test.png')






if __name__ == "__main__":
    # model_list = ['TNG50-1', 'TNG100-1', 'illustris-1']
    # save_tsne_results(model_list)

    # plot_tsne(key='illustris')
    # plot_tsne(key='illustris', include_sdss=True)
    # plot_tsne(key='nihao')
    # plot_tsne(key='nihao', include_sdss=True)
    # plot_compare()

    center, major_axis_length, minor_axis_length, angle = find_cover_range(key='illustris')
    main('illustris', center, major_axis_length, minor_axis_length, angle)


