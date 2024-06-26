import numpy as np
import scipy
from PIL import Image
import os
import shutil
import random
import math
import pickle
import pandas as pd


def copy_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    files = os.listdir(source_dir)

    for idx, file_name in enumerate(files, start=1):
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        shutil.copy2(source_file, destination_file)

    print(f"source dir: {source_dir}")
    print(f"destination dir: {destination_dir}")
    print(f"{idx} files copied successfully!")



def count_files(dir):
    files = os.listdir(dir)
    num_files = len(files)

    return num_files



def pixel_value(image_path):
    image = Image.open(image_path)
    color_mode = image.mode
    print("Color Mode:", color_mode)

    image_array = np.array(image)
    print("Shape of the array:", image_array.shape)
    r_channel = image_array[:, :, 0]
    max_r_value = np.max(r_channel)
    print("Max value of r channel:", max_r_value)



def rgba2rgb(model_str_list):
    subsets = ['data/mock_test', 'data/mock_train']
    for subset in subsets:
        for model_str in model_str_list:
            image_dir = os.path.join(subset, model_str, 'test')
            for filename in os.listdir(image_dir):
                image_path = os.path.join(image_dir, filename)
                img = Image.open(image_path)
                rgb_img = img.convert('RGB')
                rgb_img.save(image_path)



def check_image_size(dir_path):
    files_in_dir = os.listdir(dir_path)
    if files_in_dir:
        random_file = random.choice(files_in_dir)
        img = Image.open(os.path.join(dir_path, random_file))
        print(f"The size of randomly sampled {random_file} from {dir_path} is {img.height} x {img.width}")
        return img.height
    else:
        print(f"The directory '{dir_path}' is empty.")



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



def sdss_split(source_folder='data/sdss/cutouts', destination_folder='data/sdss_data'):
    os.makedirs(os.path.join(destination_folder, "train", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "esval", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "val", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "test", 'cutouts'), exist_ok=True)

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
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "train", 'cutouts', file_))

    for file_ in esval_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "esval", 'cutouts', file_))

    for file_ in val_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "val", 'cutouts', file_))

    for file_ in test_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "test", 'cutouts', file_))



def oversample_minority(source_folder, destination_folder, repeat):
    os.makedirs(destination_folder, exist_ok=True)

    for image in os.listdir(source_folder):
        image_path = os.path.join(source_folder, image)
        filename, extension = os.path.splitext(image)

        for i in range(repeat):
            new_filename = f"{filename}_copy{i}{extension}"
            new_image_path = os.path.join(destination_folder, new_filename)
            shutil.copy(image_path, new_image_path)

    # print(f"{source_folder}: {len(os.listdir(source_folder))}")
    # print(f"{destination_folder}: {len(os.listdir(destination_folder))}")



def add_subdir_move_files(base_dir, new_dir):
    path = os.path.join(base_dir, new_dir)

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created.")
    else:
        print(f"'{path}' already exists within '{base_dir}'.")

    for filename in os.listdir(base_dir):
        source_file = os.path.join(base_dir, filename)
        if os.path.isdir(source_file):
            continue
        destination_file = os.path.join(path, filename)
        shutil.move(source_file, destination_file)



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



def copy_df_path_images(df_dir, destination_dir, model_str):
    df = pd.read_pickle(os.path.join(df_dir, model_str + '.pkl'))

    for index, row in df.iterrows():
        filename_full_path = row['filename']

        base_name = os.path.basename(filename_full_path)
        destination_path = os.path.join(destination_dir, base_name)

        shutil.copy2(filename_full_path, destination_path)



if __name__ == '__main__':
    pass
    # source_dir = "data/sdss_data/val/cutouts"
    # destination_dir = "data/sdss/cutouts"
    # copy_files(source_dir, destination_dir)
    # print(count_files(source_dir))
    # print(count_files(destination_dir))

    # pixel_value("data/sdss_data/test/cutouts/16.png")
    # pixel_value("../mock-images/illustris/illustris-1_snapnum_135/broadband_1.png")
    # pixel_value("../mock-images/NIHAO/faceon_1114/bh_faceon_g1.05e11.png")
    # pixel_value("../mock-images/NIHAOrt/mockobs_1012/AGN_g1.05e13_00.png")
