import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import shutil

import src.main.classifier as classifier 


def data_prepare(savepath_prefix, prefix, nz, undersample_list, oversample_list, original_list, frac, random_state, key=True):
    new_dir = os.path.join(prefix, 'latent-vectors')
    os.makedirs(new_dir, exist_ok=True)

    # undersampling TNG100
    for model_str in undersample_list:
        majority_df = pd.read_pickle(os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', 'inlier', model_str + '.pkl'))
        undersampled_majority_df = majority_df.sample(frac=frac, random_state=random_state)
        undersampled_majority_df.pop('mahalanobis')
        # print(undersampled_majority_df.head(1))
        undersampled_majority_df.to_pickle(os.path.join(new_dir, model_str + '.pkl'))

    if key:
        for model_str in original_list:
            from_path = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors', model_str + '.pkl')
            to_path = os.path.join(new_dir, model_str + '.pkl')
            shutil.copy2(from_path, to_path)
    else:
        for model_str in original_list:
            model_df = pd.read_pickle(os.path.join(savepath_prefix, 'outlier-detect', 'in-out-sep', 'train', 'inlier', model_str + '.pkl'))
            model_df.pop('mahalanobis')
            model_df.to_pickle(os.path.join(new_dir, model_str + '.pkl'))

    for model_str in oversample_list:
        from_path = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors', model_str + '.pkl')
        # print(pd.read_pickle(from_path).head(1))
        to_path = os.path.join(new_dir, model_str + '.pkl')
        shutil.copy2(from_path, to_path)

    for f in os.listdir(new_dir):
        if f.endswith('.pkl'):
            current_df = pd.read_pickle(os.path.join(new_dir, f))
            with open(os.path.join(new_dir, 'print-message.txt'), "a") as text_file:
                text_file.write(f"{f}: {current_df.shape} \n\n")



if __name__ == '__main__':
    cuda_num = '0'
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    max_iter = 400
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    frac = 0.2
    random_state = 0
    prefix = os.path.join(savepath_prefix, 'undersampling-test', 'test_frac_5_random_0') # manually
    
    # undersample_list = ['TNG100']
    # oversample_list = ['UHDrt', 'n80rt']
    # original_list = ['AGNrt', 'NOAGNrt', 'TNG50']
    # data_prepare(savepath_prefix, prefix, nz, undersample_list, oversample_list, original_list, frac, random_state)

    # classifier.make_directory(prefix)

    load_data_dir = os.path.join(prefix, 'latent-vectors')
    save_dir = os.path.join(prefix, 'classification')
    # classifier.cross_val(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir)

    classifier.classify(savepath_prefix, prefix, nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir)
    