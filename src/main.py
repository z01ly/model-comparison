import os
import numpy as np
import pandas as pd
import torch

import src.pre

import src.infoVAE.utils
import src.infoVAE.mmdVAE_test


class ModelComparison():
    def __init__(self, model_str_list, image_size, z_dim):
        self.model_str_list = model_str_list
        self.image_size = image_size # 64: src.pre.check_image_size('data/sdss_data/test/cutouts')
        self.z_dim = z_dim # 32

    def __call__(self):
        self.image_pre()
        self.infovae_encode() # parameters

    # step 1
    def image_pre(self):
        for model_str in self.model_str_list:
            mock_img_path = os.path.join('data', model_str)

            # check size of mock images
            mock_data_size = src.pre.check_image_size(mock_img_path)

            # upsample or downsample mock images if the size doesn't match sdss size
            if mock_data_size != self.image_size:
                src.pre.sample_mock(mock_img_path)
            # split mock images to training set and test set
            src.pre.mock_split(mock_img_path, model_str)

            # add a subdir named 'test' to prepare the directory for infoVAE dataloader
            src.pre.add_subdir_move_files(os.path.join('data/mock_train/', model_str),  'test')
            src.pre.add_subdir_move_files(os.path.join('data/mock_test/', model_str), 'test')

    # step 2
    def infovae_encode(self, gpu_id=1, workers=4, batch_size=500, nc=3, use_cuda=True):
        n_filters = self.image_size
        after_conv = src.infoVAE.utils.conv_size_comp(self.image_size)

        # infoVAE model
        vae = src.infoVAE.mmdVAE_train.Model(self.z_dim, nc, n_filters, after_conv)
        vae.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))
        if use_cuda:
            vae = vae.cuda(gpu_id)
        vae.eval()

        # encode images to latent vectors
        with torch.no_grad():
            for key in ['train', 'test']:
                for model_str in self.model_str_list:
                    mock_dataroot = os.path.join('data/mock_' + key, model_str)

                    z, filename_arr = src.infoVAE.mmdVAE_test.test_with_filename(vae, test_dataroot=mock_dataroot, 
                        use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)

                    z_filename_df = pd.DataFrame(z, columns=[f'f{i}' for i in range(self.z_dim)])
                    # filename example: data/mock_train/UHDrt/test/UHD_g6.96e11_06.png
                    z_filename_df.insert(self.z_dim, "filename", filename_arr, allow_duplicates=False)
                    
                    z_filename_df.to_pickle(os.path.join('src/results/latent-vectors/' + key, model_str + '.pkl'))

    # step 3
    def outlier_detect_m(self):
        pass
    
    



if __name__ == '__main__':
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    image_size = 64
    z_dim = 32
    mc = ModelComparison(model_str_list, image_size, z_dim)
    mc.infovae_encode()