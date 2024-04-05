import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import LabelEncoder

import src.classification.utils as utils
import src.infoVAE.mmdVAE_test
import src.pre


def reencode_sdss_test(vae_save_path, image_size=64, z_dim=32, gpu_id=0, workers=4, batch_size=500, nc=3, use_cuda=True):
    mock_dataroot_dir = 'data/sdss_data'
    to_pickle_dir = 'src/results/latent-vectors/sdss'
    os.makedirs(to_pickle_dir, exist_ok=True)

    src.infoVAE.mmdVAE_test.test_main(['test'], vae_save_path, mock_dataroot_dir, to_pickle_dir, 
    gpu_id, workers, batch_size, image_size, nc, z_dim, n_filters=image_size, use_cuda=use_cuda)


def map_model_idx(model_str_list, print_key=False):
    label_binarizer = LabelEncoder().fit(model_str_list)
    model_idx_dict = {}
    for class_label, int_label in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        model_idx_dict[class_label] = int_label
        if print_key:
            print(f"{class_label}: {model_idx_dict[class_label]}")

    return model_idx_dict


def filter_df(input_df, indices, dest_dir, pkl_name):
    df = input_df.iloc[indices]
    df.reset_index(drop=True, inplace=True)
    df.to_pickle(os.path.join(dest_dir, pkl_name + '.pkl'))


# plot some example images that have high probability and some that have low probability
def example_img(low, high, model_str, model_str_list, save_dir, key, classifier_key, sdss_test_df, z_dim):
    os.makedirs(os.path.join('src/results/classification-inlier/example-sdss', model_str), exist_ok=True)
    clf = pickle.load(open(os.path.join(save_dir, 'save-model', key, classifier_key + '-model.pickle'), "rb"))
    sdss_test_data = sdss_test_df.iloc[:, 0:z_dim].to_numpy()
    model_idx_dict = map_model_idx(model_str_list)

    sdss_pred_prob = clf.predict_proba(sdss_test_data)
    current_col = sdss_pred_prob[:, model_idx_dict[model_str]]

    low_idx = np.where(current_col < low)[0]
    # low_arr = current_col[current_col < low]
    print(len(low_idx))

    high_idx = np.where(current_col > high)[0]
    # high_arr= current_col[current_col > high]
    print(len(high_idx))

    filter_df(sdss_test_df, low_idx, os.path.join('src/results/classification-inlier/example-sdss', model_str), 'low')
    source_dir = os.path.join('src/results/classification-inlier/example-sdss', model_str)
    destination_dir = os.path.join(source_dir, 'low')
    os.makedirs(destination_dir, exist_ok=True)
    src.pre.copy_df_path_images(source_dir, destination_dir, 'low')

    filter_df(sdss_test_df, high_idx, os.path.join('src/results/classification-inlier/example-sdss', model_str), 'high')
    destination_dir = os.path.join(source_dir, 'high')
    os.makedirs(destination_dir, exist_ok=True)
    src.pre.copy_df_path_images(source_dir, destination_dir, 'high')




if __name__ == "__main__":
    # vae_save_path = 'src/infoVAE/mmdVAE_save/checkpoint.pt'
    # reencode_sdss_test(vae_save_path, image_size=64, z_dim=32, gpu_id=0, workers=4, batch_size=500, nc=3, use_cuda=True)
    
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    sdss_test_df = pd.read_pickle('src/results/latent-vectors/sdss/test.pkl')
    save_dir = 'src/results/classification-inlier'
    example_img(0.1, 0.99, 'TNG100', model_str_list, save_dir, 'NIHAOrt_TNG', 'stacking-MLP-RF-XGB', sdss_test_df, 32)
    