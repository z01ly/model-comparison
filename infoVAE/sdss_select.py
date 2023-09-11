import numpy as np
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def apply_tsne_once():
    pass



def save_tsne_results():
    filename_latent_list = ['AGN.npy', 'NOAGN.npy', 'UHD.npy', 'n80.npy']
    for filename_latent in filename_latent_list:
        latent_z = np.load('./test_results/latent/' + filename_latent) 
        z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)
        np.save('./tsne/tsne_embedded_' + filename_latent, z_embedded)

    with open('./test_results/latent/sdss_test_with_filename.pkl', 'rb') as pickle_file:
        combined_data_dict = pickle.load(pickle_file)

    sdss_latent = combined_data_dict['codes']
    z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(sdss_latent)
    np.save('./tsne/tsne_embedded_sdss_test.npy', z_embedded)




def plot_tsne(include_oversample=False, include_sdss=False):
    fig = plt.figure(figsize=(10,10))

    colors = ['b', 'r', 'g', 'k', 'y']
    markers = ['o', 'v', 's', '*', '+']
    i = 0
    for filename_latent in os.listdir('./test_results/latent/'):
        
        if (not include_sdss) and (filename_latent == 'sdss_test.npy'):
            continue

        if (not include_oversample) and ((filename_latent == 'UHD_2times.npy') or (filename_latent == 'n80_2times.npy')):
            continue
        elif include_oversample and ((filename_latent == 'UHD.npy') or (filename_latent == 'n80.npy')):
            continue
        

        latent_z = np.load('./test_results/latent/' + filename_latent) # e.g. shape of UHD: (126, 32)
        z_embedded = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)


        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_latent[: -4])

        i += 1

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Applying tsne to latent z")
    plt.legend()

    
    if include_sdss:
        fig.savefig('./tsne/plot_tsne_all.png')
    else:
        if include_oversample:
            fig.savefig('./tsne/plot_tsne_models_oversample.png')
        else:
            fig.savefig('./tsne/plot_tsne_models.png')



def find_cover_range():
    fig = plt.figure(figsize=(10,10))

    colors = ['b', 'r', 'g', 'k', 'y']
    markers = ['o', 'v', 's', '*', '+']
    i = 0
    global_max_distance = 0
    filename_latent_list = ['AGN.npy', 'NOAGN.npy', 'UHD.npy', 'n80.npy']
    for filename_latent in filename_latent_list:
        # latent_z = np.load('./test_results/latent/' + filename_latent) 
        z_embedded = np.load('./tsne/tsne_embedded_' + filename_latent)
        # np.save('./tsne/tsne_embedded_' + filename_latent, z_embedded)

        euclidean_distances = np.linalg.norm(z_embedded, axis=1)
        local_max = np.max(euclidean_distances)
        if local_max > global_max_distance:
            global_max_distance = local_max

        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_latent[: -4])
        i += 1

    selected_circle = plt.Circle((0, 0), global_max_distance - 2.5, fill=False, color=colors[i])
    max_dist_circle = plt.Circle((0, 0), global_max_distance, fill=False, color='c')
    fig.gca().add_artist(selected_circle)
    fig.gca().add_artist(max_dist_circle)
    plt.xlim(-int(global_max_distance + 2), int(global_max_distance + 2))
    plt.ylim(-int(global_max_distance + 2), int(global_max_distance + 2))

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Applying tsne to latent z")
    plt.legend()

    fig.savefig('./tsne/find_cover_range.png')

    return global_max_distance, global_max_distance - 2.5



def main(global_max_distance, selected_distance):
    with open('./test_results/latent/sdss_test_with_filename.pkl', 'rb') as pickle_file:
        combined_data_dict = pickle.load(pickle_file)

    # sdss_latent = combined_data_dict['codes']
    # sdss_filenames = combined_data_dict['filenames']

    sdss_embedded = np.load('./tsne/tsne_embedded_sdss_test.npy')
    # np.save('./tsne/tsne_embedded_sdss_test.npy', sdss_embedded)

    distances = np.linalg.norm(sdss_embedded, axis=1)
    indices = np.where(distances <= selected_distance)[0]

    selected_sdss_embedded = sdss_embedded[indices]
    # selected_sdss_latent = sdss_latent[indices]
    # selected_sdss_filenames = sdss_filenames[indices]
    # np.savetxt('./tsne/selected_sdss_filenames.txt', selected_sdss_filenames, fmt="%s")

    fig = plt.figure(figsize=(10,10))
    colors = ['b', 'r', 'g', 'k']
    markers = ['o', 'v', 's', '*']
    i = 0
    filename_latent_list = ['AGN.npy', 'NOAGN.npy', 'UHD.npy', 'n80.npy']
    for filename_latent in filename_latent_list:
        z_embedded = np.load('./tsne/tsne_embedded_' + filename_latent)
        plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_latent[: -4])
        i += 1

    plt.scatter(selected_sdss_embedded[:, 0], selected_sdss_embedded[:, 1], s=5, c='y', marker='+', label='selected sdss test')

    selected_circle = plt.Circle((0, 0), selected_distance, fill=False, color='m')
    max_dist_circle = plt.Circle((0, 0), global_max_distance, fill=False, color='c')
    fig.gca().add_artist(selected_circle)
    fig.gca().add_artist(max_dist_circle)
    plt.xlim(-int(global_max_distance + 2), int(global_max_distance + 2))
    plt.ylim(-int(global_max_distance + 2), int(global_max_distance + 2))

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title("Selected sdss test")
    plt.legend()

    fig.savefig('./tsne/select_sdss_test.png')





if __name__ == "__main__":
    # plot_tsne(False, False)
    # plot_tsne(True, False)

    global_max_distance, selected_distance = find_cover_range()
    main(global_max_distance, selected_distance)
