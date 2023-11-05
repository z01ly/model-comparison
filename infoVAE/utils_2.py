import numpy as np
import scipy
from PIL import Image
import os
import shutil
import random
import math
import pickle



def sdss_size():
    folder_path = '../sdss_data/test/cutouts'
    for filename in os.listdir(folder_path):
        filepath = f"{folder_path}/{filename}"
        img = Image.open(filepath)
        break

    print("{} x {}".format(img.height, img.width))



def mock_images_size():
    filepaths = ["../NOAGN/test/classic_faceon_g1.05e11.png",
        "../AGN/test/bh_g1.18e10_08.png",
        "../n80/test/n80_faceon_g2.57e11_n80.0_e0.13.png",
        "../UHD/test/UHD_1.12e12_06.png",
        "../mockobs_0915/test/g1.08e11_01.png"]

    for filepath in filepaths:
        img = Image.open(filepath)
        print("{} x {} \n".format(img.height, img.width))



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




if __name__ == '__main__':
    # sdss_size() 64 x 64
    # mock_images_size() 500 x 500 -> 64 x 64

    # sample_mock('../NOAGN/test')
    # sample_mock('../AGN/test')
    # sample_mock('../n80/test')
    # sample_mock('../UHD/test')
    # sample_mock('../mockobs_0915/test')

    sample_mock('../illustris-1_snapnum_135')
    sample_mock('../TNG100-1_snapnum_099')
    sample_mock('../TNG50-1_snapnum_099')

    # sdss_split()

    """
    source_folder = '../UHD/test'
    destination_folder = '../UHD_2times/test'
    oversample_minority(source_folder, destination_folder, 2)

    source_folder = '../n80/test'
    destination_folder = '../n80_2times/test'
    oversample_minority(source_folder, destination_folder, 2)
    """
