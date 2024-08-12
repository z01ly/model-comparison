import torch

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy

 
def check_range(savepath_prefix, nz, model_str_list):
    model_str_list = model_str_list.copy()
    model_str_list.append('sdss-test')

    for model_str in model_str_list:
        if model_str == 'sdss-test':
            df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl'))
        else:
            train_test_dfs = []
            for key in ['train', 'test']:
                z_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', key, model_str + '.pkl'))
                train_test_dfs.append(z_df)

            df = pd.concat(train_test_dfs, axis=0)
            df.reset_index(drop=True, inplace=True)

        np_arr = df.iloc[:, 0:nz].to_numpy()

        with open(os.path.join(savepath_prefix, 'vis', 'latent-space', 'range-txt', model_str + '.txt'), "w") as text_file:
            text_file.write(f"global min: {np.min(np_arr)}, global max: {np.max(np_arr)} \n")
            text_file.write(f"\n")

        for i in range(nz):
            with open(os.path.join(savepath_prefix, 'vis', 'latent-space', 'range-txt', model_str + '.txt'), "a") as text_file:
                text_file.write(f"dim {i} \n")
                text_file.write(f"min: {np.min(np_arr[:, i])}, max: {np.max(np_arr[:, i])}, average: {np.mean(np_arr[:, i])} \n")



def main(savepath_prefix, max_point, nz, model_str, vae, gpu_id, use_cuda=True):
    z_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', 'train', model_str + '.pkl'))
    np_arr = z_df.iloc[:, 0:nz].to_numpy()

    sampled_idx = np.random.choice(np_arr.shape[0], size=1, replace=False)
    sampled_vec = np_arr[sampled_idx].squeeze()

    start_range = - max_point
    end_range = max_point

    for j in range(nz):
        vec = copy.deepcopy(sampled_vec)

        fig, axes = plt.subplots(1, 15, figsize=(25, 3))

        # points = np.append(np.linspace(start_range, end_range, num=14), vec[j])
        points = np.linspace(start_range, end_range, num=14)
        pos_to_insert = np.searchsorted(points, vec[j])
        points = np.insert(points, pos_to_insert, vec[j])

        for itr, (point, ax) in enumerate(zip(points, axes)):
            vec[j] = point
            gen_z = torch.from_numpy(vec).unsqueeze(0)
            gen_z.requires_grad_ = False
            if use_cuda:
                gen_z = gen_z.cuda(gpu_id)
            reconstructed_img = vae.decoder(gen_z)

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

        savefig_path = os.path.join(savepath_prefix, 'vis', 'latent-space', 'dim-example', model_str, f'dim{j}.png')
        plt.savefig(savefig_path, dpi=300)
        plt.close(fig)

