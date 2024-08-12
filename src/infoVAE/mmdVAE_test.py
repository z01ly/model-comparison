import torch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE import Model
from src.infoVAE.utils import apply_k_sparse, test_file_save_df


def test_with_filename(model, test_dataroot, config, use_cuda=True):
    # dataloader with filename, e.g. data/mock_train/UHDrt/test/UHD_g6.96e11_06.png
    dataloader = utils.dataloader_func(test_dataroot,
                                       config['data_params']['test_batch_size'],
                                       config['data_params']['num_workers'],
                                       True,
                                       with_filename=True)

    z_sparse_list = []
    z_dense_list = []
    filename_list = []
    for batch_idx, (test_x, _, img_filenames) in enumerate(dataloader):
        test_x.requires_grad_(False)
        if (use_cuda):
            test_x = test_x.cuda(config['trainer_params']['gpu_id'])

        _, _, _, _, z = model(test_x)
        z_sparse = apply_k_sparse(z, k_pre=int(config['model_params']['latent_dim'] * 0.0625), alpha=2)

        z_sparse_list.append(z_sparse.cpu().data.numpy())
        z_dense_list.append(z.cpu().data.numpy())
        filename_list.extend(img_filenames)
    
    # z[idx] corresponds to filename_arr[idx]
    z_sparse = np.concatenate(z_sparse_list, axis=0)
    z_dense = np.concatenate(z_dense_list, axis=0)
    filename_arr = np.asarray(filename_list)

    return z_sparse, z_dense, filename_arr



def test_main(model_str_list, vae_save_path, mock_dataroot_dir, to_pickle_dir, dense_pickle_dir):
    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # infoVAE model
    vae = Model(config['model_params']['latent_dim'], config['model_params']['in_channels'])
    vae.load_state_dict(torch.load(vae_save_path))
    vae = vae.cuda(config['trainer_params']['gpu_id'])
    vae.eval()

    # encode images to latent vectors
    with torch.no_grad():
        for model_str in model_str_list:
            mock_dataroot = os.path.join(mock_dataroot_dir, model_str)

            z_sparse, z_dense, filename_arr = test_with_filename(vae, test_dataroot=mock_dataroot, config=config, use_cuda=True)

            test_file_save_df(config, z_sparse, filename_arr.copy(), model_str, to_pickle_dir)
            test_file_save_df(config, z_dense, filename_arr, model_str, dense_pickle_dir)




if __name__ == "__main__":
    gpu_id = 1
    workers = 4
    batch_size = 500 # CUDA out of memory if 500x500 image size
    image_size = 64 # downsample 500x500 test images to 64x64
    nc = 3
    nz = 32 # Size of z latent vector
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)
    use_cuda = True

    # model
    model = Model(nz, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

    with torch.no_grad():
        # test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in os.listdir('src/data/mock_trainset')]
        # test_dataroots.extend([os.path.join('src/data/mock_valset', subdir) for subdir in os.listdir('src/data/mock_valset')])

        # test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in ['UHD_2times', 'n80_2times', 'TNG50-1_snapnum_099_2times']]
        test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in ['mockobs_0915_2times']]
        for test_dataroot in test_dataroots:
            directory_names = test_dataroot.split(os.path.sep)
            extraction = f"{directory_names[-2][5: ]}_{directory_names[-1]}"

            savefig_path = 'src/infoVAE/test_results/images_in_testing/fig_' + extraction + '.png'
            z = test_with_filename(model, test_dataroot=test_dataroot, savefig_path=savefig_path,
                    z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                    use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)

            np.save('src/infoVAE/test_results/latent/' + extraction + '.npy', z)

