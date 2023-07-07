import torch
from torchvision import transforms, datasets
import numpy as np
from PIL import Image

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
# 500 x 500
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
# 500 x 500
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
# 500 x 500
# n80_size()

def UHD_size():
    filepath_1 = "../UHD/test/UHD_1.12e12_06.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../UHD/test/UHD_faceon_2.79e12.png"
    img_2 = Image.open(filepath_2)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
# 500 x 500
# UHD_size()

def no_FB_size():
    filepath_1 = "../no_FB/test/no_FB_g1.50e10_04.png"
    img_1 = Image.open(filepath_1)

    print("{} x {} \n".format(img_1.height, img_1.width))
# 500 x 500
# no_FB_size()
