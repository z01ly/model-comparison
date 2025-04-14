import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.infoVAE.mmdVAE import Model
from src.infoVAE.utils import conv_size_comp


def explained_var(savepath_prefix, z_dim):
    os.makedirs(os.path.join(savepath_prefix, 'pca-test'), exist_ok=True)

    sdss_test_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')) 
    sdss_test_data = sdss_test_df.iloc[:, 0:z_dim].to_numpy()
    sc = StandardScaler()
    sdss_test_data = sc.fit_transform(sdss_test_data)

    dim_list = []
    sum_explained_var_ratio_list = []
    for n_components in range(1, z_dim+1):
        dim_list.append(n_components)
        pca = PCA(n_components=n_components)
        pca.fit(sdss_test_data)
        explained_variance_ratio_ = pca.explained_variance_ratio_
        sum_explained_var_ratio_list.append(np.sum(explained_variance_ratio_))

    fig = plt.figure(figsize=(12, 6))
    plt.plot(dim_list, sum_explained_var_ratio_list)
    plt.xticks(range(1, z_dim+2))
    plt.yticks([i/10 for i in range(11)])
    plt.ylabel('Sum of Explained variance Ratio')
    plt.xlabel('Number of Principal components')
    plt.title('Explained Variance Ratio vs. Number of PCA Components')
    plt.grid(True)
    # plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(os.path.join(savepath_prefix, 'pca-test', 'explained_var_ratio.png'), bbox_inches='tight')



class MoveAlongAxis():
    def __init__(self, savepath_prefix, nz, n_components, key, display_list):
        # key = 'inlier' or 'outlier'
        self.savepath_prefix = savepath_prefix
        self.nz = nz
        self.n_components = n_components
        self.key = key
        self.display_list = display_list


    def load_data(self):
        directory = os.path.join(self.savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', self.key)
        dfs = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            df = pd.read_pickle(filepath)
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=0)
        combined_df.reset_index(drop=True, inplace=True)
        del combined_df['filename']
        del combined_df['mahalanobis']

        combined_np_arr = combined_df.to_numpy()

        return combined_np_arr


    def load_vae(self, gpu_id, use_cuda=True):
        image_size = 64
        after_conv = conv_size_comp(image_size)

        vae = Model(z_dim=self.nz, nc=3, n_filters=64, after_conv=after_conv)
        vae.load_state_dict(torch.load(os.path.join(self.savepath_prefix, 'infoVAE', 'checkpoint.pt')))
        if use_cuda:
            vae = vae.cuda(gpu_id)
        vae.eval()

        return vae


    def move_along_pca_axis(self, principal_component, latent_mean, vae, display_num, gpu_id, use_cuda=True):
        step_size = 1
        num_steps = 10
        images = []
        for step in range(-num_steps, num_steps):
            new_latent = latent_mean + step * step_size * principal_component
            with torch.no_grad():
                gen_z = torch.from_numpy(new_latent).unsqueeze(0)
                gen_z.requires_grad_ = False
                if use_cuda:
                    gen_z = gen_z.cuda(gpu_id)
                reconstructed_img = vae.decoder(gen_z)
                reconstructed_array = reconstructed_img.contiguous().cpu().data.numpy()
                reconstructed_array = reconstructed_array.squeeze().transpose(1, 2, 0)
                images.append(reconstructed_array)

        return images

    
    def plot_images(self, principal_component, latent_mean, vae, display_num, gpu_id):
        images = self.move_along_pca_axis(principal_component, latent_mean, vae, display_num, gpu_id)

        fig, axes = plt.subplots(1, len(images), figsize=(20, 2))

        for img, ax in zip(images, axes):
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(savepath_prefix, 'pca-test', 'move-along-axis', f'{display_num}', self.key + '.png'), dpi=300)
        plt.close(fig)


    def main(self, gpu_id):
        vae = self.load_vae(gpu_id)

        data_arr = self.load_data()
        data_arr = StandardScaler().fit_transform(data_arr)
        latent_mean = np.mean(data_arr, axis=0)

        pca = PCA(n_components=self.n_components)
        pca.fit(data_arr)

        for display_num in self.display_list:
            os.makedirs(os.path.join(savepath_prefix, 'pca-test', 'move-along-axis', f'{display_num}'), exist_ok=True)
            principal_component = pca.components_[display_num]
            self.plot_images(principal_component, latent_mean, vae, display_num, gpu_id)




if __name__ == "__main__":
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    gpu_id = 4

    # explained_var(savepath_prefix, nz)

    inlier = MoveAlongAxis(savepath_prefix, nz, 3, 'inlier', [0, 1, 2])
    inlier.main(gpu_id)
    outlier = MoveAlongAxis(savepath_prefix, nz, 3, 'outlier', [0, 1, 2])
    outlier.main(gpu_id)
