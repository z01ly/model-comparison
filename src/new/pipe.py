import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import yaml

import src.main.oversampling as oversampling
import src.main.classifier as classifier 
from src.new.utils import df_to_gen_score
from src.classification.example_sdss import filter_df
import src.classification.train_test_API as train_test_API
import src.classification.train_test_tree as train_test_tree
from src.pre import copy_df_path_images
import src.main.img_encoder as img_encoder
import src.main.latent_vis as latent_vis


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

    message_dir = save_dir
    classifier.classifier_train(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir, message_dir)



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



class GenOod():
    def __init__(self, c_str, savepath_prefix, sdss_dir, id_dir, gamma=0.1, M=6):
        # sdss_dir = os.path.join(savepath_prefix, 'classification')
        # id_dir = os.path.join(savepath_prefix, 'classify-ID')
        self.c_str = c_str
        self.savepath_prefix = savepath_prefix
        self.sdss_dir = sdss_dir
        self.id_dir = id_dir

        self.gamma = gamma
        self.M = M


    def plot(self):
        os.makedirs(os.path.join(self.savepath_prefix, 'gen-ood', 'plot'), exist_ok=True)

        sdss_negative_scores = df_to_gen_score(self.sdss_dir, self.c_str, self.gamma, self.M)
        ID_negative_scores = df_to_gen_score(self.id_dir, self.c_str, self.gamma, self.M)

        sns.histplot(sdss_negative_scores, bins=50, kde=True, stat='density', label='sdss')
        sns.histplot(ID_negative_scores, bins=50, kde=True, stat='density', label='sim-test (ID)')
        plt.legend()
        plt.xlabel('Negative score')
        plt.ylabel('Density')
        plt.title('Distribution of negative GEN scores')

        plt.savefig(os.path.join(self.savepath_prefix, 'gen-ood', 'plot', self.c_str + '.png'))
        plt.close()


    def select_sdss(self, id_percent): 
        os.makedirs(os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-vectors', self.c_str), exist_ok=True)

        sdss_negative_scores = df_to_gen_score(self.sdss_dir, self.c_str, self.gamma, self.M)
        ID_negative_scores = df_to_gen_score(self.id_dir, self.c_str, self.gamma, self.M)

        ID_threshold = np.percentile(ID_negative_scores, id_percent)
        # print(ID_threshold)

        sdss_test_df_path = os.path.join(self.savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
        sdss_test_df = pd.read_pickle(sdss_test_df_path)

        sdss_id_idx = np.where(sdss_negative_scores > ID_threshold)[0]
        print("ID indices shape: ", sdss_id_idx.shape)
        sdss_ood_idx = np.setdiff1d(np.arange(sdss_test_df.shape[0]), sdss_id_idx)
        print("OOD indices shape: ", sdss_ood_idx.shape)

        filter_df(sdss_test_df, sdss_id_idx, os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-vectors', self.c_str), 'id')
        filter_df(sdss_test_df, sdss_ood_idx, os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-vectors', self.c_str), 'ood')


    def re_classify(self, model_str_list, nz):
        os.makedirs(os.path.join(self.savepath_prefix, 'gen-ood', 're-classify'), exist_ok=True)

        test_save_dir = os.path.join(self.savepath_prefix, 'gen-ood', 're-classify')
        for directory in ['prob-df', 'violin-plot']:
            os.makedirs(os.path.join(test_save_dir, directory), exist_ok=True)

        sdss_test_df_path = os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-vectors', self.c_str, 'id.pkl')
        sdss_test_df = pd.read_pickle(sdss_test_df_path)
        sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()
        print("Selected SDSS test data shape: ", sdss_test_data.shape)
        
        if self.c_str in ['random-forest', 'xgboost']:
            train_test_tree.test(self.sdss_dir, test_save_dir, model_str_list, self.c_str, sdss_test_data)
        elif self.c_str in ['stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']:
            train_test_API.test(self.sdss_dir, test_save_dir, model_str_list, self.c_str, sdss_test_data)

    
    def copy_sdss_imgs(self):
        df_dir = os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-vectors', self.c_str)
        for i in ['id', 'ood']:
            dest_dir = os.path.join(self.savepath_prefix, 'gen-ood', 'selected', 'sdss-imgs', self.c_str, i)
            os.makedirs(dest_dir, exist_ok=True)
            copy_df_path_images(df_dir, dest_dir, i)




if __name__ == "__main__":
    gpu_id = 6
    nz = 32
    savepath_prefix = 'new-sparse'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # classifiers = ['random-forest', 'xgboost', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']

    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # infoVAE func
    # img_encoder.vae(savepath_prefix)
    # img_encoder.plot_training(savepath_prefix, 24, 0.0015, 0.0015)
    # img_encoder.plot_residual(savepath_prefix, config, model_str_list, use_cuda=True)
    # img_encoder.encoder(savepath_prefix, model_str_list)
    

    # latent vis func
    # model_str_dict = {'AGNrt': 0.9, 'NOAGNrt': 0.9, 'TNG100': 0.8, 'TNG50': 0.9, 'UHDrt': 1.0, 'n80rt': 1.0}
    # latent_vis.tsne_vis(savepath_prefix, config['model_params']['latent_dim'], model_str_dict, model_str_list)
    latent_vis.latent_space_vis(savepath_prefix, config, model_str_list, use_cuda=True)


    # minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']
    # oversample_sim(savepath_prefix, nz, model_str_list, minority_str_list, gpu_id)


    # cuda_num = str(gpu_id)
    # max_iter = 600
    # classify_calibration_train(savepath_prefix, nz, model_str_list, cuda_num, max_iter)


    # classify_test(savepath_prefix, nz, model_str_list)


    # classify_ID_test(savepath_prefix, nz, model_str_list)


    # sdss_dir = os.path.join(savepath_prefix, 'classification')
    # id_dir = os.path.join(savepath_prefix, 'classify-ID')
    # gen = GenOod('stacking-MLP-RF-XGB', savepath_prefix, sdss_dir, id_dir)
    # gen.select_sdss(5)
    # gen.re_classify(model_str_list, nz)
    # gen.copy_sdss_imgs()
