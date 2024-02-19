# This file is from https://github.com/CKleiber/SciML-Seminar
# Some parameters are modified

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torch
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.infoVAE.utils
from src.infoVAE.mmdVAE_train import Model


def decode_latent_space(model, latent_vector):
    model.eval()
    latent_vector = torch.from_numpy(latent_vector).float()
    latent_vector = latent_vector.unsqueeze(0)
    decoded_image = model.decoder(latent_vector)
    decoded_image = decoded_image.detach().numpy()
    decoded_image = decoded_image.squeeze(0)
    decoded_image = decoded_image.transpose(1, 2, 0)
    decoded_image = decoded_image * 255
    decoded_image = decoded_image.astype(np.uint8)
    return decoded_image


class LatentSpaceExplorer:
    def __init__(self, model, latent_dim, latent_vector, rows, cols):
        self.latent_dim = latent_dim
        self.latent_vector = latent_vector # np.zeros(latent_dim)
        self.model = model
        self.rows = rows
        self.cols = cols

        self.root = tk.Tk()
        self.root.title("Latent Space Explorer")

        self.create_gui()

    def create_gui(self):
        # Create sliders for each dimension organized in a 5x10 grid
        # rows = 10
        # cols = 5
        self.sliders = []
        for i in range(self.rows):
            for j in range(self.cols):
                index = i * self.cols + j
                slider = tk.Scale(self.root, from_=-5, to=5, orient="horizontal", length=200,
                                   command=lambda val, index=index: self.update_latent_vector(val, index),
                                   resolution=0.01, label=f"{index}")
                slider.grid(row=i, column=j, padx=5, pady=5)
                slider.set(self.latent_vector[index])
                self.sliders.append(slider)

        # Create canvas for displaying decoded image with increased size
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.grid(row=0, column=self.cols, rowspan=self.rows, padx=10, pady=10)

        # Initial update of the canvas
        self.update_canvas()

    def update_latent_vector(self, value, index):
        self.latent_vector[index] = float(value)
        self.update_canvas()

    def update_canvas(self):
        decoded_image = decode_latent_space(self.model, self.latent_vector)
        self.display_image(decoded_image)

    def display_image(self, image_array):
        # Convert NumPy array to PhotoImage
        img = Image.fromarray(image_array.astype('uint8'))
        img = ImageTk.PhotoImage(img.resize((500, 500), Image.LANCZOS))

        # Update canvas with the new image
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img

    def run(self):
        self.root.mainloop()



if __name__ == '__main__':
    workers = 4
    image_size = 64
    nc = 3
    nz = 32
    n_filters = 64
    after_conv = src.infoVAE.utils.conv_size_comp(image_size)

    # model
    vae = Model(nz, nc, n_filters, after_conv)
    vae.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt', map_location ='cpu'))
    vae.eval()

    with torch.no_grad():
        # Use one vector from UHD as an example
        # UHD_train_inlier = pd.read_pickle('src/results/latent-vectors/train-inlier-original/UHDrt.pkl').iloc[:, 0:nz].to_numpy()
        # latent_vector = UHD_train_inlier[0]
        latent_vector = np.array([0.05171053, 0.06476861, -0.11704127, 0.01460991, 0.10246621, -0.00567089, 0.01452429, 
                                  -0.12356801, -0.10543184, -0.00155619, 0.11496778, -0.05734455, -0.00591347, -0.01004576,
                                  -0.24052174, -0.10696202, 0.16584918, -0.00324588, -0.1083241, -0.04741355, 0.06078181,
                                  -0.00824846, 0.05670254, -0.07864703, 0.149401, 0.02224097, -0.18136238, 0.09258986,
                                  -0.08658882, -0.00267665, 0.19264245, 0.12813681])
        explorer = LatentSpaceExplorer(vae, nz, latent_vector, 8, 4)
        explorer.run()