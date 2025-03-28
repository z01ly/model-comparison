import os
import random
import shutil

import numpy as np
import scipy
from PIL import Image
import src.data.utils

import src.config as config


# ====================================
# Part 1: IllustrisTNG Processing
# ====================================




# =======================================
# Part 2: Simulation Data Processing
# =======================================

def mock_split(source_directory, model_str, rate=0.85):
    train_dir = 'data/mock_train/' + model_str
    test_dir = 'data/mock_test/' + model_str

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files_in_source_directory = os.listdir(source_directory)

    total_files = len(files_in_source_directory)
    train_count = int(total_files * rate)
    test_count = total_files - train_count

    random.shuffle(files_in_source_directory)

    train_files = files_in_source_directory[:train_count]
    test_files = files_in_source_directory[train_count:]

    for file in train_files:
        source_path = os.path.join(source_directory, file)
        dest_path = os.path.join(train_dir, file)
        shutil.copy2(source_path, dest_path)

    for file in test_files:
        source_path = os.path.join(source_directory, file)
        dest_path = os.path.join(test_dir, file)
        shutil.copy2(source_path, dest_path)

    print(f"current model: {model_str}")
    print(f"{len(os.listdir(train_dir))} files copied to the training set.")
    print(f"{len(os.listdir(test_dir))} files copied to the test set.")


def sample_mock(folder_path): # for upsampling and downsampling
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        image_array = np.array(image)

        # sample the image to 64x64 using scipy.ndimage.zoom
        downsampled_array = scipy.ndimage.zoom(image_array, (64 / image_array.shape[0], 64 / image_array.shape[1], 1), order=3)

        downsampled_image = Image.fromarray(downsampled_array.astype(np.uint8))

        # Save the sampled image, overwriting the original file
        downsampled_image.save(image_path)





def mock_data_pre(model_str_list, image_size):
    for model_str in model_str_list:
        mock_img_path = os.path.join('data', model_str)

        # check size of mock images
        mock_data_size = src.data.utils.check_image_size(mock_img_path)

        # upsample or downsample mock images if the size doesn't match sdss size
        if mock_data_size != image_size:
            src.data.utils.sample_mock(mock_img_path)
        # split mock images to training set and test set
        src.data.utils.mock_split(mock_img_path, model_str)

        # add a subdir named 'test' to prepare the directory for infoVAE dataloader
        src.data.utils.add_subdir_move_files(os.path.join('data/mock_train/', model_str), 'test')
        src.data.utils.add_subdir_move_files(os.path.join('data/mock_test/', model_str), 'test')

    src.data.utils.pixel_value('data/mock_test/AGNrt/test/AGN_g1.05e13_10.png')
    src.data.utils.pixel_value('data/mock_train/TNG100/test/broadband_1.png')

    src.data.utils.rgba2rgb(model_str_list)



def mock_data_count(model_str_list):
    with open(os.path.join('results', 'data-details.txt'), "a") as f:
        f.write(f"\n\nmock data:\n")
        for model_str in model_str_list:
            data_dir = os.path.join('data', model_str)
            num_files = src.data.utils.count_files(data_dir)
            f.write(f"{num_files} files in {data_dir}. \n")

        f.write(f"\nTraining and test set:\n")
        for model_str in model_str_list:
            for key in ['mock_train', 'mock_test']:
                data_dir = os.path.join('data', key, model_str, 'test')
                num_files = src.data.utils.count_files(data_dir)
                f.write(f"{num_files} files in {data_dir}. \n")

    


# =============================
# Part 3: SDSS Processing
# =============================

def sdss_split(source_folder=config.SDSS_CUTOUTS_PATH, destination_folder=config.SDSS_IMAGE_PATH):
    """Split SDSS images to train, validation and test set"""
    os.makedirs(config.SDSS_TRAIN_PATH, exist_ok=True)
    os.makedirs(config.SDSS_ESVAL_PATH, exist_ok=True)
    os.makedirs(config.SDSS_VAL_PATH, exist_ok=True)
    os.makedirs(config.SDSS_TEST_PATH, exist_ok=True)

    image_files = os.listdir(source_folder)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_ratio = 0.7
    esval_ratio = 0.1
    val_ratio = 0.1
    # test_ratio = 0.1
    num_train = int(total_images * train_ratio)
    num_esval = int(total_images * esval_ratio)
    num_val = int(total_images * val_ratio)
    num_test = total_images - num_train - num_esval - num_val

    train_files = image_files[: num_train]
    esval_files = image_files[num_train: num_train + num_esval]
    val_files = image_files[num_train + num_esval: num_train + num_esval + num_val]
    test_files = image_files[num_train + num_esval + num_val: ]

    for file_ in train_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_TRAIN_PATH, file_))

    for file_ in esval_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_ESVAL_PATH, file_))

    for file_ in val_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_VAL_PATH, file_))

    for file_ in test_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_TEST_PATH, file_))




if __name__ == '__main__':
    # sdss_split()
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # mock_data_pre(model_str_list, 64)
    mock_data_count(model_str_list)
