import os
import numpy as np
import pandas as pd

import src.pre


def sdss_count():
    s = 0
    for subset in ['train', 'esval', 'val', 'test']:
        data_dir = os.path.join('data/sdss_data', subset, 'cutouts')
        num_files = src.pre.count_files(data_dir)
        s += num_files
        with open(os.path.join('results', 'data-details.txt'), "a") as f:
            f.write(f"{num_files} files in {data_dir}. \n")

    with open(os.path.join('results', 'data-details.txt'), "a") as f:
        f.write(f"sdss: {s} files in total. \n")



def mock_data_pre(model_str_list, image_size):
    for model_str in model_str_list:
        mock_img_path = os.path.join('data', model_str)

        # check size of mock images
        mock_data_size = src.pre.check_image_size(mock_img_path)

        # upsample or downsample mock images if the size doesn't match sdss size
        if mock_data_size != image_size:
            src.pre.sample_mock(mock_img_path)
        # split mock images to training set and test set
        src.pre.mock_split(mock_img_path, model_str)

        # add a subdir named 'test' to prepare the directory for infoVAE dataloader
        src.pre.add_subdir_move_files(os.path.join('data/mock_train/', model_str), 'test')
        src.pre.add_subdir_move_files(os.path.join('data/mock_test/', model_str), 'test')

    src.pre.pixel_value('data/mock_test/AGNrt/test/AGN_g1.05e13_10.png')
    src.pre.pixel_value('data/mock_train/TNG100/test/broadband_1.png')

    src.pre.rgba2rgb(model_str_list)



def mock_data_count(model_str_list):
    with open(os.path.join('results', 'data-details.txt'), "a") as f:
        f.write(f"\n\nmock data:\n")
        for model_str in model_str_list:
            data_dir = os.path.join('data', model_str)
            num_files = src.pre.count_files(data_dir)
            f.write(f"{num_files} files in {data_dir}. \n")

        f.write(f"\nTraining and test set:\n")
        for model_str in model_str_list:
            for key in ['mock_train', 'mock_test']:
                data_dir = os.path.join('data', key, model_str, 'test')
                num_files = src.pre.count_files(data_dir)
                f.write(f"{num_files} files in {data_dir}. \n")

    


if __name__ == '__main__':
    # src.pre.sdss_split()
    sdss_count()
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # mock_data_pre(model_str_list, 64)
    mock_data_count(model_str_list)
