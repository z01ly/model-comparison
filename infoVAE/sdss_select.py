import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def apply_tsne_once():
    pass



def save_tsne_results():
    filename_latent_list = ['AGN.npy', 'NOAGN.npy', 'UHD.npy', 'n80.npy', 'UHD_2times.npy', \
        'n80_2times.npy', 'sdss_test.npy', 'mockobs_0915.npy']
    for filename_latent in filename_latent_list:
        latent_z = np.load('./test_results/latent/' + filename_latent)
        z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)
        np.save('./tsne/tsne_' + filename_latent, z_embedded)



def plot_tsne(include_oversample=False, include_sdss=False):
    fig = plt.figure(figsize=(10,10))

    colors = ['b', 'r', 'g', 'k', 'y', 'm']
    markers = ['o', 'v', 's', '*', '+', 'h']
    i = 0
    filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy', 'tsne_UHD_2times.npy', \
        'tsne_n80_2times.npy', 'tsne_sdss_test.npy', 'tsne_mockobs_0915.npy']
    for filename_embedded in filename_embedded_list:
        if (not include_sdss) and (filename_embedded == 'tsne_sdss_test.npy'):
            continue

        if (not include_oversample) and ((filename_embedded == 'tsne_UHD_2times.npy') or (filename_embedded == 'tsne_n80_2times.npy')):
            continue
        elif include_oversample and ((filename_embedded == 'tsne_UHD.npy') or (filename_embedded == 'tsne_n80.npy')):
            continue
        
        z_embedded = np.load('./tsne/' + filename_embedded)

        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])

        i += 1

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Applying tsne to latent z")
    plt.legend()

    
    if include_sdss:
        fig.savefig('./tsne/plot_all.png')
    else:
        if include_oversample:
            fig.savefig('./tsne/plot_models_oversample.png')
        else:
            fig.savefig('./tsne/plot_models.png')



def find_cover_range():
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)

    colors = ['b', 'r', 'g', 'k', 'y']
    markers = ['o', 'v', 's', '*', '+']
    i = 0
    filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy']
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

    """
    ellipse2 = Ellipse(
        xy=center,
        width=major_axis_length + 5,
        height=minor_axis_length + 5,
        angle=np.degrees(angle),
        edgecolor='m',
        facecolor='none'
    )
    ax.add_patch(ellipse2)
    """
    
    ax.set_aspect('equal', adjustable='box') # Set aspect ratio to equal
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Fitting an ellipse to tsne data")
    plt.legend()

    fig.savefig('./tsne/find_cover_range.png')

    return center, major_axis_length, minor_axis_length, angle




def main(center, major_axis_length, minor_axis_length, angle):
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
    np.save('./test_results/latent/selected_sdss_test.npy', selected_sdss_latent)
    np.savetxt('./test_results/latent_txt/selected_sdss_test.txt', selected_sdss_latent, delimiter=',', fmt='%s')
    selected_sdss_filenames = sdss_filenames[indices]
    np.savetxt('./tsne/selected_sdss_filenames.txt', selected_sdss_filenames, fmt="%s")

    fig = plt.figure(figsize=(10,10))
    colors = ['b', 'r', 'g', 'k']
    markers = ['o', 'v', 's', '*']
    i = 0
    filename_embedded_list = ['tsne_AGN.npy', 'tsne_NOAGN.npy', 'tsne_UHD.npy', 'tsne_n80.npy']
    for filename_embedded in filename_embedded_list:
        z_embedded = np.load('./tsne/' + filename_embedded)
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_embedded[: -4])
        i += 1

    plt.scatter(selected_sdss_embedded[:, 0], selected_sdss_embedded[:, 1], s=5, c='y', marker='+', label='selected sdss test')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Selected sdss test")
    plt.legend()

    fig.savefig('./tsne/select_sdss_test.png')





if __name__ == "__main__":
    save_tsne_results()

    plot_tsne(False, False)
    plot_tsne(True, False)
    plot_tsne(False, True)

    # center, major_axis_length, minor_axis_length, angle = find_cover_range()
    # main(center, major_axis_length, minor_axis_length, angle)


