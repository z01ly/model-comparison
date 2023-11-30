import numpy as np
import scipy
from PIL import Image
import os
import shutil
import random
import math
import pickle


def check_image_size(dir_path):
    files_in_dir = os.listdir(dir_path)
    if files_in_dir:
        random_file = random.choice(files_in_dir)
        img = Image.open(os.path.join(dir_path, random_file))

        print(f"The size of randomly sampled file {random_file} is {img.height} x {img.width}")
    else:
        print(f"The directory '{dir_path}' is empty.")

    return img.height



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



def sdss_split():
    source_folder = '../sdss/cutouts'
    destination_folder = '../sdss_data'

    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "train", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "val", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "test", 'cutouts'), exist_ok=True)

    image_files = os.listdir(source_folder)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_ratio = 0.7
    val_ratio = 0.2
    num_train = int(total_images * train_ratio)
    num_val = int(total_images * val_ratio)
    num_test = total_images - num_train - num_val

    train_files = image_files[: num_train]
    val_files = image_files[num_train: num_train + num_val]
    test_files = image_files[num_train + num_val: ]

    for file_ in train_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "train", 'cutouts', file_))

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

    print(f"{source_folder}: {len(os.listdir(source_folder))}")
    print(f"{destination_folder}: {len(os.listdir(destination_folder))}")



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



def mock_split(source_directory, rate=0.85):
    train_directory = 'src/data/mock_trainset/' + source_directory[9: ] # exclude 'src/data/'
    val_directory = 'src/data/mock_valset/' + source_directory[9: ]

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(val_directory, exist_ok=True)

    files_in_source_directory = os.listdir(source_directory)

    total_files = len(files_in_source_directory)
    train_count = int(total_files * rate)
    val_count = total_files - train_count

    random.shuffle(files_in_source_directory)

    train_files = files_in_source_directory[:train_count]
    val_files = files_in_source_directory[train_count:]

    for file in train_files:
        source_path = os.path.join(source_directory, file)
        dest_path = os.path.join(train_directory, file)
        shutil.copy2(source_path, dest_path)

    for file in val_files:
        source_path = os.path.join(source_directory, file)
        dest_path = os.path.join(val_directory, file)
        shutil.copy2(source_path, dest_path)

    print(f"current model: {source_directory[9: ]}")
    print(f"{len(os.listdir(train_directory))} files copied to the training set.")
    print(f"{len(os.listdir(val_directory))} files copied to the validation set.")





if __name__ == '__main__':
    pass

    # sdss_split()
