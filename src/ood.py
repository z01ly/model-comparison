import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import softmax

import src.classification.train_test_API as train_test_API
import src.classification.train_test_tree as train_test_tree
from src.data.utils import copy_df_path_images, filter_df
import src.config as config



def gen_ood(softmax_id_val, gamma, M):
    """
    Generalized entropy score
    source: https://github.com/XixiLiu95/GEN/blob/main/benchmark.py
    """
    probs = softmax_id_val
    probs_sorted = np.sort(probs, axis=1)[:,-M:]
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
        
    return -scores 



def df_to_gen_score(save_dir, c, gamma, M):
    prob_df = pd.read_pickle(os.path.join(save_dir, 'prob-df', c + '.pkl'))
    probs = prob_df.to_numpy()

    softmax_probs = softmax(probs, axis=1)

    negative_scores = gen_ood(softmax_probs, gamma, M)

    return negative_scores




class GenOod():
    def __init__(self,
                 c_str,
                 sdss_dir,
                 id_dir,
                 savepath=config.RESULTS_GEN_OOD,
                 gamma=0.1,
                 M=6):
        self.c_str = c_str
        self.savepath = savepath
        self.sdss_dir = sdss_dir
        self.id_dir = id_dir

        self.gamma = gamma
        self.M = M


    def plot(self, percent_p, x1, x2, y, legend_loc):
        os.makedirs(os.path.join(self.savepath, 'plot'), exist_ok=True)

        sdss_negative_scores = df_to_gen_score(self.sdss_dir, self.c_str, self.gamma, self.M)
        ID_negative_scores = df_to_gen_score(self.id_dir, self.c_str, self.gamma, self.M)

        ID_threshold = np.percentile(ID_negative_scores, percent_p)
        print(f"ID_threshold: {ID_threshold}")

        sns.histplot(sdss_negative_scores, bins=50, kde=True, stat='density', label='sdss')
        sns.histplot(ID_negative_scores, bins=50, kde=True, stat='density', label='sim-test (ID)')

        plt.axvline(x=ID_threshold, color='r', linestyle='--', linewidth=2)

        plt.text(x1, y, 'OOD', color='r', fontsize=16, ha='center', va='center')
        plt.text(x2, y, 'ID', color='r', fontsize=16, ha='center', va='center')

        # plt.legend()
        plt.xlabel('Negative score', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.title('Distribution of negative GEN scores', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.legend(handletextpad=1, markerscale=2, fontsize=16, loc=legend_loc)

        plt.savefig(os.path.join(self.savepath, 'plot', self.c_str + '.png'))
        plt.close()


    def select_sdss(self, percent_p, sdss_save_prefix=config.RESULTS_PATH):
        percent_dir = os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-vectors', self.c_str)
        os.makedirs(percent_dir, exist_ok=True)

        sdss_negative_scores = df_to_gen_score(self.sdss_dir, self.c_str, self.gamma, self.M)
        ID_negative_scores = df_to_gen_score(self.id_dir, self.c_str, self.gamma, self.M)

        ID_threshold = np.percentile(ID_negative_scores, percent_p)
        # print(ID_threshold)

        sdss_test_df_path = os.path.join(sdss_save_prefix, 'latent-vectors', 'sdss', 'test.pkl')
        sdss_test_df = pd.read_pickle(sdss_test_df_path)

        sdss_id_idx = np.where(sdss_negative_scores > ID_threshold)[0]
        print("ID indices shape: ", sdss_id_idx.shape)
        sdss_ood_idx = np.setdiff1d(np.arange(sdss_test_df.shape[0]), sdss_id_idx)
        print("OOD indices shape: ", sdss_ood_idx.shape)

        filter_df(sdss_test_df, sdss_id_idx, percent_dir, 'id')
        filter_df(sdss_test_df, sdss_ood_idx, percent_dir, 'ood')


    def re_classify(self, model_str_list, nz, percent_p):
        test_save_dir = os.path.join(self.savepath, 're-classify', f"percent{percent_p}")
        os.makedirs(test_save_dir, exist_ok=True)
        for j in ['prob-df', 'violin-plot']:
            os.makedirs(os.path.join(test_save_dir, j), exist_ok=True)

        sdss_id_df_path = os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-vectors', self.c_str, 'id.pkl')
        sdss_id_df = pd.read_pickle(sdss_id_df_path)
        sdss_id_data = sdss_id_df.iloc[:, 0:nz].to_numpy()
        print("Selected SDSS test data shape: ", sdss_id_data.shape)

        if self.c_str in ['random-forest', 'xgboost']:
            train_test_tree.test(self.sdss_dir, test_save_dir, model_str_list, self.c_str, sdss_id_data)
        elif self.c_str in ['stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']:
            train_test_API.test(self.sdss_dir, test_save_dir, model_str_list, self.c_str, sdss_id_data)


    def copy_sdss_imgs(self, percent_p):
        percent_dir = os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-vectors', self.c_str)
        for i in ['id', 'ood']:
            dest_dir = os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-imgs', self.c_str, i)
            os.makedirs(dest_dir, exist_ok=True)
            copy_df_path_images(percent_dir, dest_dir, i)


    def print_message(self, percent_p):
        percent_dir = os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-vectors', self.c_str)
        sdss_test_id = pd.read_pickle(os.path.join(percent_dir, 'id.pkl'))
        id_row, id_column = sdss_test_id.shape
        sdss_test_ood = pd.read_pickle(os.path.join(percent_dir, 'ood.pkl'))
        ood_row, ood_column = sdss_test_ood.shape
        # print(id_column, ood_column)
        total_row = id_row + ood_row

        with open(os.path.join(self.savepath, 'selected', f"percent{percent_p}", 'sdss-id-ood-ratio.txt'), "a") as f:
            f.write(f"classifier: {self.c_str} \n")
            f.write(f"ID number: {id_row}, OOD number: {ood_row}, total number: {total_row}\n")
            f.write(f"id ratio: {id_row / total_row}\n")
            f.write(f"ood ratio: {ood_row / total_row}\n\n")

