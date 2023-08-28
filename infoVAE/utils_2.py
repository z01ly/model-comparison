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



def NOAGN_size():
    filepath_1 = "../NOAGN/test/classic_faceon_g1.05e11.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../NOAGN/test/classic_g1.08e11_10.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../NOAGN/test/ell_wobh_faceon_g1.27e12.png"
    img_3 = Image.open(filepath_3)

    filepath_4 = "../NOAGN/test/ell_wobh_g1.17e13_01.png"
    img_4 = Image.open(filepath_4)


    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))
    print("{} x {} \n".format(img_4.height, img_4.width))



def AGN_size():
    filepath_1 = "../AGN/test/bh_faceon_g1.05e11.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../AGN/test/bh_g1.18e10_08.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../AGN/test/ell_bh_faceon_g6.53e12.png"
    img_3 = Image.open(filepath_3)

    filepath_4 = "../AGN/test/ell_bh_g1.14e13_01.png"
    img_4 = Image.open(filepath_4)


    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))
    print("{} x {} \n".format(img_4.height, img_4.width))



def n80_size():
    filepath_1 = "../n80/test/g1.37e11_n80.0_e0.13_15.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../n80/test/g7.66e11_n80.0_e0.13_Cstar0.13_06.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../n80/test/n80_faceon_g2.57e11_n80.0_e0.13.png"
    img_3 = Image.open(filepath_3)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))



def UHD_size():
    filepath_1 = "../UHD/test/UHD_1.12e12_06.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../UHD/test/UHD_faceon_2.79e12.png"
    img_2 = Image.open(filepath_2)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))



def downsample_mock(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        image_array = np.array(image)

        # Downsample the image to 64x64 using scipy.ndimage.zoom
        downsampled_array = scipy.ndimage.zoom(image_array, (64 / image_array.shape[0], 64 / image_array.shape[1], 1), order=3)

        downsampled_image = Image.fromarray(downsampled_array.astype(np.uint8))

        # Save the downsampled image, overwriting the original file
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
    # NOAGN_size() 500 x 500 -> 64 x 64
    # AGN_size() 500 x 500 -> 64 x 64
    # n80_size() 500 x 500 -> 64 x 64
    # UHD_size() 500 x 500 -> 64 x 64

    # downsample_mock('../NOAGN/test')
    # downsample_mock('../AGN/test')
    # downsample_mock('../n80/test')
    # downsample_mock('../UHD/test')

    # sdss_split()

    # source_folder = '../UHD/test'
    # destination_folder = '../UHD_10times/test'
    # oversample_minority(source_folder, destination_folder, 10)

    # source_folder = '../n80/test'
    # destination_folder = '../n80_5times/test'
    # oversample_minority(source_folder, destination_folder, 5)
