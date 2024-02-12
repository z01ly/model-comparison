import src.infoVAE.utils
from src.infoVAE.mmdVAE_train import Model
from PIL import Image

import torch
from torch.autograd import Variable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy

 
def check_range(model_str):
    input_data = src.infoVAE.utils.stack_train_val(model_str)

    check_dir = os.path.join("src/dim/dim-meaning", model_str)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    else:
        print(f"Directory already exists: {check_dir}")

    with open(os.path.join("src/dim/dim-meaning", model_str, "range.txt"), "w") as text_file:
        text_file.write(f"global min: {np.min(input_data)}, global max: {np.max(input_data)} \n")
        text_file.write(f"\n")


    for i in range(32):
        with open(os.path.join("src/dim/dim-meaning", model_str, "range.txt"), "a") as text_file:
                text_file.write(f"dim {i} \n")
                text_file.write(f"min: {np.min(input_data[:, i])}, max: {np.max(input_data[:, i])}, average: {np.mean(input_data[:, i])} \n")




def main(model_str, model, gpu_id, use_cuda=True, dimension=32):
    input_data = src.infoVAE.utils.stack_train_val(model_str)

    sampled_indices = np.random.choice(input_data.shape[0], size=5, replace=False)
    sampled_vectors = input_data[sampled_indices] # shape: (5, dimension)

    for i in range(5):
        os.makedirs(os.path.join("src/dim/dim-meaning", model_str, f"vector_{i}"), exist_ok=True)
        vec_original = sampled_vectors[i]

        start_range = -5
        end_range = 5

        for j in range(dimension):
            vec = copy.deepcopy(vec_original)

            fig, axes = plt.subplots(1, 15, figsize=(25, 3))

            # points = np.append(np.linspace(start_range, end_range, num=14), vec[j])
            points = np.linspace(start_range, end_range, num=14)
            pos_to_insert = np.searchsorted(points, vec[j])
            points = np.insert(points, pos_to_insert, vec[j])

            for itr, (point, ax) in enumerate(zip(points, axes)):
                vec[j] = point
                gen_z = Variable(torch.from_numpy(vec).unsqueeze(0), requires_grad=False)
                if use_cuda:
                    gen_z = gen_z.cuda(gpu_id)
                reconstructed_img = model.decoder(gen_z)

                reconstructed_array = reconstructed_img.contiguous().cpu().data.numpy()
                reconstructed_array = reconstructed_array.squeeze().transpose(1, 2, 0)
                # reconstructed_array = (reconstructed_array * 255).astype(int)

                ax.imshow(reconstructed_array)
                if itr == pos_to_insert:
                    ax.set_title(f'point {point:.2f}', color='red')
                else:
                    ax.set_title(f'point {point:.2f}')
                ax.axis('off')
                
            plt.tight_layout()

            savefig_path = os.path.join("src/dim/dim-meaning", model_str, f"vector_{i}", f"dim{j}.png")
            plt.savefig(savefig_path, dpi=300)

            plt.close(fig)
 

        


if __name__ == '__main__':
    model_str_list = ['TNG50-1', 'TNG100-1', 'illustris-1']
    # for model_str in model_str_list:
    #     check_range(model_str)

    gpu_id = 2
    image_size = 64
    nc = 3
    nz = 32
    z_dim = nz
    n_filters = 64
    after_conv = src.infoVAE.utils.conv_size_comp(image_size)
    use_cuda = True

    model = Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

    with torch.no_grad():
        for model_str in model_str_list:
            main(model_str, model, gpu_id, use_cuda=True, dimension=32)

