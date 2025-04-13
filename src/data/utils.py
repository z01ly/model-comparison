import numpy as np
from PIL import Image
import os
import shutil
import random
import pandas as pd
from typing import List

import src.config as config


def copy_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    files = os.listdir(source_dir)

    for idx, file_name in enumerate(files, start=1):
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        shutil.copy2(source_file, destination_file)

    print(f"Source dir: {source_dir}")
    print(f"Destination dir: {destination_dir}")
    print(f"{idx} files copied successfully!")



def count_files(dir):
    files = os.listdir(dir)
    num_files = len(files)

    return num_files



def sdss_count():
    s = 0
    for subset in ['train', 'esval', 'val', 'test']:
        data_dir = os.path.join(config.SDSS_IMAGE_PATH, subset, 'cutouts')
        num_files = count_files(data_dir)
        s += num_files
        with open(os.path.join(config.RESULTS_PATH, 'data-details.txt'), "a") as f:
            f.write(f"{num_files} files in {data_dir}. \n")

    with open(os.path.join(config.RESULTS_PATH, 'data-details.txt'), "a") as f:
        f.write(f"SDSS: {s} files in total. \n")



def mock_data_count(model_str_list): # TODO
    with open(os.path.join('results', 'data-details.txt'), "a") as f:
        f.write(f"\n\nmock data:\n")
        for model_str in model_str_list:
            data_dir = os.path.join('data', model_str)
            num_files = count_files(data_dir)
            f.write(f"{num_files} files in {data_dir}. \n")

        f.write(f"\nTraining and test set:\n")
        for model_str in model_str_list:
            for key in ['mock_train', 'mock_test']:
                data_dir = os.path.join('data', key, model_str, 'test')
                num_files = count_files(data_dir)
                f.write(f"{num_files} files in {data_dir}. \n")



def check_color_mode(image_path: str) -> None:
    image = Image.open(image_path)
    color_mode = image.mode
    print("Color mode:", color_mode)

    image_array = np.array(image)
    print("Shape of the array:", image_array.shape)
    # r_channel = image_array[:, :, 0]
    # max_r_value = np.max(r_channel)
    # print("Max value of r channel:", max_r_value)
    return



def rgba2rgb(model_str_list: List[str], mock_train_or_test_path: str) -> None: 
    """
    Convert RGBA image to RGB mode
    Used in combination with cubic_sampling(folder_path: str)
    Run check_color_mode(image_path: str) before applying this function
    """
    for model_str in model_str_list:
        image_dir = os.path.join(mock_train_or_test_path, model_str, 'class_0')
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
        print(f"Size of randomly sampled {random_file} from {dir_path}: {img.height} x {img.width}")
        return img.height
    else:
        print(f"The directory '{dir_path}' is empty.")



# def add_subdir_move_files(base_dir, new_dir):
#     path = os.path.join(base_dir, new_dir)
# 
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#         print(f"Directory '{path}' created.")
#     else:
#         print(f"'{path}' already exists within '{base_dir}'.")
# 
#     for filename in os.listdir(base_dir):
#         source_file = os.path.join(base_dir, filename)
#         if os.path.isdir(source_file):
#             continue
#         destination_file = os.path.join(path, filename)
#         shutil.move(source_file, destination_file)



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

    # check_color_mode("data/sdss-data/test/cutouts/16.png")
    check_color_mode('data/processed/sampling2/train/AGNrt/class_0/AGN_g1.14e13_11.png')
    check_color_mode('data/processed/sampling2/train/TNG100/class_0/broadband_41600.png')
    check_color_mode('data/sampled/sampling2/NIHAOrt/noAGN_g1.27e12_00.png')