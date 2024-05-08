import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

import src.pre 
import src.infoVAE.mmdVAE_test as mmdVAE_test


def img_copy(savepath_prefix, model_str_list):
    for model_str in model_str_list:
        os.makedirs(os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images', model_str), exist_ok=True)

    df_dir = os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', 'inlier')

    for model_str in model_str_list:
        destination_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images', model_str)
        src.pre.copy_df_path_images(df_dir, destination_dir, model_str)



def img_oversample(savepath_prefix, model_str_list, minority_str_list):
    images_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images')
    oversampled_images_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-images')
    for model_str in model_str_list:
        os.makedirs(os.path.join(oversampled_images_dir, model_str), exist_ok=True)

    for minority_str in minority_str_list:
        src.pre.oversample_minority(os.path.join(images_dir, minority_str), 
                                    os.path.join(oversampled_images_dir, minority_str), 
                                    2)
    
    for model_str in (np.setdiff1d(model_str_list, minority_str_list)):
        src.pre.oversample_minority(os.path.join(images_dir, model_str), 
                                    os.path.join(oversampled_images_dir, model_str), 
                                    1)

    for model_str in model_str_list:
        src.pre.add_subdir_move_files(os.path.join(oversampled_images_dir, model_str), 'test')



def infovae_reencode(savepath_prefix, model_str_list, gpu_id, nz):
    workers = 4
    batch_size = 500
    image_size = 64
    nc = 3
    n_filters = 64
    vae_save_path = os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')

    os.makedirs(os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors'), exist_ok=True)

    mock_dataroot_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-images')
    to_pickle_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors')
    mmdVAE_test.test_main(model_str_list, vae_save_path, mock_dataroot_dir, to_pickle_dir, 
    gpu_id, workers, batch_size, image_size, nc, nz, n_filters=image_size, use_cuda=True)



def print_messages(savepath_prefix, model_str_list):
    with open(os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'print-message.txt'), "w") as txt_file:
        txt_file.write(f"Number of images: \n\n")
        for model_str in model_str_list:
            num_files = src.pre.count_files(os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images', model_str))
            txt_file.write(f"{model_str}: {num_files} images copied. \n")

        txt_file.write(f"\nOversampling: \n\n")
        for model_str in model_str_list:
            num_files = src.pre.count_files(os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-images', model_str, 'test'))
            txt_file.write(f"{model_str}: {num_files} images after oversampling. \n")



if __name__ == '__main__':
    gpu_id = 5
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']

    # img_copy(savepath_prefix, model_str_list)
    # img_oversample(savepath_prefix, model_str_list, minority_str_list)
    # print_messages(savepath_prefix, model_str_list)
    infovae_reencode(savepath_prefix, model_str_list, gpu_id, nz)
