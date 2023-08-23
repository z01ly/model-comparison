import utils
from mmdVAE_train import Model
from PIL import Image

import torch
from torch.autograd import Variable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os



# random sample 10 latent vectors from each model and sdss_test
# for each latent vector, 
#   for each dimension of 32 dims,
#       change it within a range and use decoder to produce image
# mmdVAE_train and plot is helpful


def load_latent_codes():
    AGN_data = np.load('./test_results/latent/AGN.npy')
    NOAGN_data = np.load('./test_results/latent/NOAGN.npy')
    UHD_data = np.load('./test_results/latent/UHD.npy')
    n80_data = np.load('./test_results/latent/n80.npy')
    sdss_test_data = np.load('../infoVAE/test_results/latent/sdss_test.npy')

    return AGN_data, NOAGN_data, UHD_data, n80_data, sdss_test_data

 
def check_range_each(input_data, current_dir):
    print(input_data.shape)

    for i in range(32):
        with open(os.path.join("./dim_meaning", current_dir, "range.txt"), "a") as text_file:
                text_file.write(f"dim {i} \n")
                text_file.write(f"min: {np.min(input_data[:, i])}, max: {np.max(input_data[:, i])}, average: {np.mean(input_data[:, i])} \n")


def check_range(AGN_data, NOAGN_data, UHD_data, n80_data, sdss_test_data):
    check_range_each(AGN_data, "AGN")
    check_range_each(NOAGN_data, "NOAGN")
    check_range_each(UHD_data, "UHD")
    check_range_each(n80_data, "n80")
    check_range_each(sdss_test_data, "sdss_test")


def main(input_data, current_dir, model, gpu_id, use_cuda=True, dimension=32):
    sampled_indices = np.random.choice(input_data.shape[0], size=10, replace=False)
    sampled_vectors = input_data[sampled_indices] # shape: (10, dimension)

    for i in range(10):
        os.makedirs(os.path.join(current_dir, f"vector_{i}"), exist_ok=True)
        vec = sampled_vectors[i]

        start_range = int(np.min(input_data) - 3)
        end_range = int(np.max(input_data) + 3)

        for j in range(dimension):
            fig, axes = plt.subplots(1, 15, figsize=(25, 3))

            points = np.append(np.linspace(start_range, end_range, num=14), vec[j])
            for point, ax in zip(points, axes):
                vec[j] = point
                gen_z = Variable(torch.from_numpy(vec).unsqueeze(0), requires_grad=False)
                if use_cuda:
                    gen_z = gen_z.cuda(gpu_id)
                reconstructed_img = model.decoder(gen_z)

                reconstructed_array = reconstructed_img.contiguous().cpu().data.numpy()
                reconstructed_array = reconstructed_array.squeeze().transpose(1, 2, 0)
                # reconstructed_array = (reconstructed_array * 255).astype(int)

                ax.imshow(reconstructed_array)
                ax.set_title(f'point {point:.2f}')
                ax.axis('off')
                
            plt.tight_layout()

            savefig_path = os.path.join(current_dir, f"vector_{i}", f"dim{j}.png")
            plt.savefig(savefig_path, dpi=300)

            plt.close(fig)


        


if __name__ == '__main__':
    AGN_data, NOAGN_data, UHD_data, n80_data, sdss_test_data = load_latent_codes()
    # check_range(AGN_data, NOAGN_data, UHD_data, n80_data, sdss_test_data)

    gpu_id = 1
    image_size = 64
    nc = 3
    nz = 32
    z_dim = nz
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)
    use_cuda = True

    model = Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('./mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

    main(AGN_data, './dim_meaning/AGN', model, gpu_id, use_cuda=True, dimension=32)
