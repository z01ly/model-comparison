import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

import src.pre 
import src.infoVAE.mmdVAE_test as mmdVAE_test


def img_copy(savepath_prefix, model_str_list, df_dir, image_dir):
    # df_dir = os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', 'inlier')

    for model_str in model_str_list:
        # current_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images', model_str)
        current_dir = os.path.join(image_dir, model_str)
        os.makedirs(current_dir, exist_ok=True)
        src.pre.copy_df_path_images(df_dir, current_dir, model_str)



def img_oversample(savepath_prefix, model_str_list, minority_str_list, image_dir, oversampled_image_dir):
    # image_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'images')
    # oversampled_image_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-images')
    for model_str in model_str_list:
        os.makedirs(os.path.join(oversampled_image_dir, model_str), exist_ok=True)

    for minority_str in minority_str_list:
        src.pre.oversample_minority(os.path.join(image_dir, minority_str), 
                                    os.path.join(oversampled_image_dir, minority_str), 
                                    2)
    
    for model_str in (np.setdiff1d(model_str_list, minority_str_list)):
        src.pre.oversample_minority(os.path.join(image_dir, model_str), 
                                    os.path.join(oversampled_image_dir, model_str), 
                                    1)

    for model_str in model_str_list:
        src.pre.add_subdir_move_files(os.path.join(oversampled_image_dir, model_str), 'test')



def infovae_reencode(savepath_prefix, model_str_list, gpu_id, nz, oversampled_image_dir, oversampled_vector_dir):
    workers = 4
    batch_size = 500
    image_size = 64
    nc = 3
    n_filters = 64
    vae_save_path = os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')

    os.makedirs(oversampled_vector_dir, exist_ok=True)

    mock_dataroot_dir = oversampled_image_dir
    to_pickle_dir = oversampled_vector_dir
    mmdVAE_test.test_main(model_str_list, vae_save_path, mock_dataroot_dir, to_pickle_dir, 
    gpu_id, workers, batch_size, image_size, nc, nz, n_filters=image_size, use_cuda=True)



def print_messages(savepath_prefix, model_str_list, base_dir):
    with open(os.path.join(base_dir, 'print-message.txt'), "w") as txt_file:
        txt_file.write(f"Number of images: \n\n")
        for model_str in model_str_list:
            num_files = src.pre.count_files(os.path.join(base_dir, 'images', model_str))
            txt_file.write(f"{model_str}: {num_files} images copied. \n")

        txt_file.write(f"\nOversampling: \n\n")
        for model_str in model_str_list:
            num_files = src.pre.count_files(os.path.join(base_dir, 'oversampled-images', model_str, 'test'))
            txt_file.write(f"{model_str}: {num_files} images after oversampling. \n")



if __name__ == '__main__':
    gpu_id = 7
    nz = 4
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']

    df_dir = os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', 'inlier')
    base_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier')
    image_dir = os.path.join(base_dir, 'images')
    oversampled_image_dir = os.path.join(base_dir, 'oversampled-images')
    oversampled_vector_dir = os.path.join(base_dir, 'oversampled-vectors')

    img_copy(savepath_prefix, model_str_list, df_dir, image_dir)
    img_oversample(savepath_prefix, model_str_list, minority_str_list, image_dir, oversampled_image_dir)
    print_messages(savepath_prefix, model_str_list, base_dir)
    infovae_reencode(savepath_prefix, model_str_list, gpu_id, nz, oversampled_image_dir, oversampled_vector_dir)
