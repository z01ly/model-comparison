import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import time
import seaborn as sns
from scipy.special import softmax

import src.main.oversampling as oversampling
import src.main.classifier as classifier 
from src.new.utils import df_to_gen_score


def oversample_sim(savepath_prefix, nz, model_str_list, minority_str_list, gpu_id):
    df_dir = os.path.join(savepath_prefix, 'latent-vectors', 'train')
    base_dir = os.path.join(savepath_prefix, 'oversampling')
    image_dir = os.path.join(base_dir, 'images')
    oversampled_image_dir = os.path.join(base_dir, 'oversampled-images')
    oversampled_vector_dir = os.path.join(base_dir, 'oversampled-vectors')

    oversampling.img_copy(savepath_prefix, model_str_list, df_dir, image_dir)
    oversampling.img_oversample(savepath_prefix, model_str_list, minority_str_list, image_dir, oversampled_image_dir)
    oversampling.print_messages(savepath_prefix, model_str_list, base_dir)
    oversampling.infovae_reencode(savepath_prefix, model_str_list, gpu_id, nz, oversampled_image_dir, oversampled_vector_dir)


def classify_calibration_train(savepath_prefix, nz, model_str_list, cuda_num, max_iter):
    classifier.make_directory(savepath_prefix)

    load_data_dir = os.path.join(savepath_prefix, 'oversampling', 'oversampled-vectors')
    save_dir = os.path.join(savepath_prefix, 'classification')
    classifier.cross_val(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir)

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    message_dir = save_dir
    classifier.classifier_train(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir, message_dir, sdss_test_df_path)


def classify_test(savepath_prefix, nz, model_str_list):
    save_dir = os.path.join(savepath_prefix, 'classification')

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    sdss_test_df = pd.read_pickle(sdss_test_df_path)

    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()
    classifier.classifier_test(save_dir, save_dir, model_str_list, sdss_test_data)


def classify_ID_test(savepath_prefix, nz, model_str_list):
    # ID == sim test set
    test_save_dir = os.path.join(savepath_prefix, 'classify-ID')
    for directory in ['prob-df', 'violin-plot']:
        os.makedirs(os.path.join(test_save_dir, directory), exist_ok=True)
    save_dir = os.path.join(savepath_prefix, 'classification')

    dfs = []
    for model_str in model_str_list:
        pkl_path = os.path.join(savepath_prefix, 'latent-vectors', 'test', model_str + '.pkl')
        df = pd.read_pickle(pkl_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0)
    combined_df.reset_index(drop=True, inplace=True)

    ID_test_data = combined_df.iloc[:, 0:nz].to_numpy()
    classifier.classifier_test(save_dir, test_save_dir, model_str_list, ID_test_data)



def gen_ood(savepath_prefix, gamma, M):
    os.makedirs(os.path.join(savepath_prefix, 'gen-ood'), exist_ok=True)

    classifiers = ['random-forest', 'xgboost', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    for c in classifiers:
        sdss_negative_scores = df_to_gen_score(os.path.join(savepath_prefix, 'classification'), c, gamma, M)
        ID_negative_scores = df_to_gen_score(os.path.join(savepath_prefix, 'classify-ID'), c, gamma, M)

        sns.histplot(sdss_negative_scores, bins=150, kde=True, stat='density', label='sdss')
        sns.histplot(ID_negative_scores, bins=150, kde=True, stat='density', label='sim-test (ID)')
        plt.legend()
        plt.xlabel('Negative score')
        plt.ylabel('Density')
        plt.title('Distribution of negative GEN scores')

        plt.savefig(os.path.join(savepath_prefix, 'gen-ood', c + '.png'))
        plt.close()

    # sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    # sdss_test_df = pd.read_pickle(sdss_test_df_path)



if __name__ == "__main__":
    gpu_id = 4
    nz = 32
    savepath_prefix = 'new'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']

    # oversample_sim(savepath_prefix, nz, model_str_list, minority_str_list, gpu_id)
    
    # cuda_num = str(gpu_id)
    # max_iter = 600
    # classify_calibration_train(savepath_prefix, nz, model_str_list, cuda_num, max_iter)

    # classify_test(savepath_prefix, nz, model_str_list)

    # classify_ID_test(savepath_prefix, nz, model_str_list)

    gen_ood(savepath_prefix, gamma=0.1, M=6)
