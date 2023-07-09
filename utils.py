import torch
from torchvision import transforms, datasets
import numpy as np
import scipy
from PIL import Image
import os

def sdss_size():
    filepath = "../cutouts_1000/cutouts_1000_train/3000.png"
    img = Image.open(filepath)

    print("{} x {}".format(img.height, img.width))
# 64 x 64
# sdss_size()

def sdss1000_load(path):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    return dataloader
# train_data = sdss1000_load('../cutouts_1000')
# test_data = sdss1000_load('../cutouts_1000_2')

def conv_size_comp(img_size):
    F = 4 # filter size
    P = 1 # padding
    S = 2 # stride
    W = (img_size - F + 2 * P) / S + 1

    return int((W - F + 2 * P) / S + 1)
# print(conv_size_comp(28))
# print(conv_size_comp(64))

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
# 500 x 500 -> 64 x 64
# NOAGN_size()

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
# 500 x 500 -> 64 x 64
# AGN_size()

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
# 500 x 500 -> 64 x 64
# n80_size()

def UHD_size():
    filepath_1 = "../UHD/test/UHD_1.12e12_06.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../UHD/test/UHD_faceon_2.79e12.png"
    img_2 = Image.open(filepath_2)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
# 500 x 500 -> 64 x 64
# UHD_size()

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
# downsample_mock('../NOAGN/test')
# downsample_mock('../AGN/test')
# downsample_mock('../n80/test')
# downsample_mock('../UHD/test')
