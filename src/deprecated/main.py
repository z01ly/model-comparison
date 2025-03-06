import os
import numpy as np
import pandas as pd
import torch

import src.pre

import src.infoVAE.utils
import src.infoVAE.mmdVAE_test

import src.outlier_detect.mahalanobis

import src.classification.utils
import src.classification.cross_val_tree
import src.classification.train_test_tree
import src.classification.cross_val_API
import src.classification.train_test_API


class ModelComparison():
    def __init__(self, model_str_list, minority_str_list, image_size, z_dim):
        self.model_str_list = model_str_list
        self.minority_str_list = minority_str_list
        self.image_size = image_size # 64: src.pre.check_image_size('data/sdss_data/test/cutouts')
        self.z_dim = z_dim # 32
        self.vae_save_path = 'src/infoVAE/mmdVAE_save/checkpoint.pt'
        self.sdss_test_data_path = 'src/results/latent-vectors/sdss_test.npy'

    def __call__(self):
        self.image_pre()
        self.infovae_encode() # parameters
        self.outlier_detect_m()
        self.imgs_copy_oversample()
        self.infovae_encode_inlier()

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
            src.pre.add_subdir_move_files(os.path.join('data/mock_train/', model_str), 'test')
            src.pre.add_subdir_move_files(os.path.join('data/mock_test/', model_str), 'test')

    # step 2
    def infovae_encode(self, gpu_id=0, workers=4, batch_size=500, nc=3, use_cuda=True):
        mock_dataroot_dir = 'data/mock_train'
        to_pickle_dir = 'src/results/latent-vectors/train'
        src.infoVAE.mmdVAE_test.test_main(self.model_str_list, self.vae_save_path, mock_dataroot_dir, to_pickle_dir, 
        gpu_id, workers, batch_size, self.image_size, nc, self.z_dim, n_filters=self.image_size, use_cuda=True)

        mock_dataroot_dir = 'data/mock_test'
        to_pickle_dir = 'src/results/latent-vectors/test'
        src.infoVAE.mmdVAE_test.test_main(self.model_str_list, self.vae_save_path, mock_dataroot_dir, to_pickle_dir, 
        gpu_id, workers, batch_size, self.image_size, nc, self.z_dim, n_filters=self.image_size, use_cuda=True)

    # step 3
    def outlier_detect_m(self, key):
        sdss_test_data = np.load(self.sdss_test_data_path)

        for model_str in self.model_str_list:
            data_df = pd.read_pickle(os.path.join('src/results/latent-vectors', key, model_str + '.pkl'))
            distance_path = os.path.join('src/results/m-distance', key, model_str + '.npy')
            mahal = src.outlier_detect.mahalanobis.MDist(model_str, self.z_dim, data_df, sdss_test_data, 
                                                        distance_path, key, alpha=0.95)
            # mahal.print_cutoff()
            mahal()

    # step 4
    def imgs_copy_oversample(self):
        for model_str in self.model_str_list:
            for key in ['inlier', 'outlier']:
                source_dir = os.path.join('src/results/latent-vectors', 'train-' + key)
                destination_dir = os.path.join('src/results/images', 'train-' + key, model_str)
                os.makedirs(destination_dir, exist_ok=True)
                src.pre.copy_df_path_images(source_dir, destination_dir, model_str)
        
        os.rename("src/results/images/train-inlier", "src/results/images/train-inlier-original")
        os.rename("src/results/latent-vectors/train-inlier", "src/results/latent-vectors/train-inlier-original")
        
        for minority_str in self.minority_str_list:
            src.pre.oversample_minority(os.path.join("src/results/images/train-inlier-original", minority_str), 
                                        os.path.join("src/results/images/train-inlier", minority_str), 
                                        2)
        
        for model_str in (np.setdiff1d(self.model_str_list, self.minority_str_list)):
            src.pre.oversample_minority(os.path.join("src/results/images/train-inlier-original", model_str), 
                                        os.path.join("src/results/images/train-inlier", model_str), 
                                        1)

        for model_str in self.model_str_list:
            src.pre.add_subdir_move_files(os.path.join("src/results/images/train-inlier", model_str), 'test')

    # step 5 
    def infovae_encode_inlier(self, gpu_id=0, workers=4, batch_size=500, nc=3, use_cuda=True):
        mock_dataroot_dir = 'src/results/images/train-inlier'
        to_pickle_dir = 'src/results/latent-vectors/train-inlier'
        os.makedirs(to_pickle_dir, exist_ok=True)

        src.infoVAE.mmdVAE_test.test_main(self.model_str_list, self.vae_save_path, mock_dataroot_dir, to_pickle_dir, 
        gpu_id, workers, batch_size, self.image_size, nc, self.z_dim, n_filters=self.image_size, use_cuda=True)

    # step 6
    def classification(self, key='NIHAOrt_TNG'):
        load_data_dir = 'src/results/latent-vectors/train-inlier'
        X, y = src.classification.utils.load_data_df(self.model_str_list, load_data_dir, self.z_dim)
        sdss_test_data = np.load(self.sdss_test_data_path)
        save_dir = 'src/results/classification-inlier'

        # cross validation
        # src.classification.cross_val_tree.main(save_dir, key, self.model_str_list, X, y, 'integer', 'random-forest')
        # src.classification.cross_val_tree.main(save_dir, key, self.model_str_list, X, y, 'integer', 'xgboost')

        # test on sdss
        # src.classification.train_test_tree.train(save_dir, key, 'random-forest', X, y)
        # src.classification.train_test_tree.train(save_dir, key, 'xgboost', X, y)
        # src.classification.train_test_tree.test(save_dir, key, self.model_str_list, 'random-forest', sdss_test_data)
        # src.classification.train_test_tree.test(save_dir, key, self.model_str_list, 'xgboost', sdss_test_data)

        classifier_keys = ['stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
        for classifier_key in classifier_keys:
            print(f"Current model: {classifier_key}")
            src.classification.cross_val_API.main(save_dir, key, self.model_str_list, X, y, classifier_key)
            print(f"Cross validation of {classifier_key} finished.")
            scaler = src.classification.train_test_API.train(save_dir, key, classifier_key, X, y)
            src.classification.train_test_API.test(save_dir, scaler, key, self.model_str_list, classifier_key, sdss_test_data)
            print(f"Train-test of {classifier_key} finished.")




if __name__ == '__main__':
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']
    image_size = 64
    z_dim = 32
    mc = ModelComparison(model_str_list, minority_str_list, image_size, z_dim)
    # mc.outlier_detect_m('test')
    mc.classification()
    