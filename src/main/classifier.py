import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import time

import src.classification.utils as utils
import src.classification.cross_val_tree as cross_val_tree
import src.classification.train_test_tree as train_test_tree
import src.classification.cross_val_API as cross_val_API
import src.classification.train_test_API as train_test_API
import src.classification.example_sdss as example_sdss


def make_directory(prefix):
    # by default: prefix = savepath_prefix

    os.makedirs(os.path.join(prefix, 'classification', 'calibration-curve'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'classification', 'confusion-matrix'), exist_ok=True)

    dirs = ['save-model', 'save-scaler', 'violin-plot']
    for dir_str in dirs:
        os.makedirs(os.path.join(prefix, 'classification', dir_str), exist_ok=True)



def cross_val(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir):
    # load_data_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors')
    # save_dir = os.path.join(savepath_prefix, 'classification')
    
    X, y = utils.load_data_df(model_str_list, load_data_dir, nz)

    classifiers = ['random-forest', 'xgboost']
    for classifier in classifiers:
        print(classifier)
        cross_val_tree.main(save_dir, model_str_list, X, y, 'integer', classifier, cuda_num)

    classifiers = ['stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    for classifier in classifiers:
        print(classifier)
        cross_val_API.main(save_dir, model_str_list, X, y, classifier, cuda_num, max_iter)



def classify(savepath_prefix, prefix, nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir):
    # by default: prefix = savepath_prefix
    with open(os.path.join(prefix, 'classification', 'print-message.txt'), "w") as text_file:
        text_file.write(f"Print training time... \n\n") 

    # load_data_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors')
    # save_dir = os.path.join(savepath_prefix, 'classification')

    X, y = utils.load_data_df(model_str_list, load_data_dir, nz)

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    sdss_test_df = pd.read_pickle(sdss_test_df_path)
    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()

    classifiers = ['random-forest', 'xgboost']
    for classifier in classifiers:
        print(classifier)

        start_time = time.time()
        train_test_tree.train(save_dir, classifier, X, y, cuda_num)
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(os.path.join(prefix, 'classification', 'print-message.txt'), "a") as text_file:
            text_file.write(f"Training time of {classifier}: {elapsed_time:.6f} seconds. \n\n") 

        train_test_tree.test(save_dir, model_str_list, classifier, sdss_test_data)

    classifiers = ['stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    for classifier in classifiers:
        print(classifier)

        start_time = time.time()
        train_test_API.train(save_dir, classifier, X, y, cuda_num, max_iter)
        end_time = time.time()
        elapsed_time = end_time - start_time
        with open(os.path.join(prefix, 'classification', 'print-message.txt'), "a") as text_file:
            text_file.write(f"Training time of {classifier}: {elapsed_time:.6f} seconds. \n\n") 

        train_test_API.test(save_dir, model_str_list, classifier, sdss_test_data)



def example_sdss_img(savepath_prefix, nz, model_str_list, classifier, model_str, low, high):
    os.makedirs(os.path.join(savepath_prefix, 'classification', 'example-sdss', model_str), exist_ok=True)
    
    save_dir = os.path.join(savepath_prefix, 'classification')
    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    sdss_test_df = pd.read_pickle(sdss_test_df_path)

    example_sdss.main(low, high, model_str, model_str_list, save_dir, classifier, sdss_test_df, nz)



if __name__ == '__main__':
    # nz = 16
    # max_iter = 450
    # nz = 20
    # max_iter = 450
    # nz = 32
    # max_iter = 300

    # nz = 2
    # max_iter = 300
    # nz = 3
    # max_iter = 400
    nz = 4
    max_iter = 500
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    cuda_num = '7'

    cross_val(savepath_prefix, nz, model_str_list, cuda_num, max_iter)
    classify(savepath_prefix, nz, model_str_list, cuda_num, max_iter)

    # for 32-dim
    # example_sdss_img(savepath_prefix, nz, model_str_list, 'stacking-MLP-RF-XGB', 'TNG100', 0.01, 0.99)
    # example_sdss_img(savepath_prefix, nz, model_str_list, 'stacking-MLP-RF-XGB', 'NOAGNrt', 0.01, 0.99)
