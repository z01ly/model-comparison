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
