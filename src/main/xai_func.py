import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from src.classification.utils import load_data_df
from src.xai.shap_plot_process import process_main
import src.xai.shap_main as shap_main


# only for random forest and xgboost
def shap_compute(prefix, load_data_dir, sdss_test_df_path, nz, model_str_list, c_str):
    os.makedirs(os.path.join(prefix, 'xai', 'shap', 'save-shap-values'), exist_ok=True)

    # sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    sdss_test_df = pd.read_pickle(sdss_test_df_path)
    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()
    sdss_test_data_sample = shap_main.test_data_sample(sdss_test_data, percent=0.6)

    # compute and save shap values
    # load_data_dir = os.path.join(savepath_prefix, 'oversampling', 'train', 'inlier', 'oversampled-vectors')
    X, y = load_data_df(model_str_list, load_data_dir, nz)
    X_sampled = shap_main.background_sample(X, y, percent=0.5)

    shap_main.save_shap_values(prefix, X_sampled, sdss_test_data_sample, c_str, 'TreeExplainer')

    
    
def shap_plot(prefix, savepath_prefix, model_str_dict, c_str, max_display):
    os.makedirs(os.path.join(prefix, 'xai', 'shap', 'beeswarm-plot', c_str), exist_ok=True)
    process_main(prefix, savepath_prefix, model_str_dict, c_str, max_display)



if __name__ == "__main__":
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'   
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    model_str_dict = {'AGNrt': 0, 'NOAGNrt': 1, 'TNG100': 2, 'TNG50': 3, 'UHDrt': 4, 'n80rt': 5} # model for plot

    # prefix = os.path.join(savepath_prefix, 'undersampling-test', 'test_frac_5_random_0')
    prefix = savepath_prefix

    # load_data_dir = os.path.join(prefix, 'latent-vectors')
    # shap_compute(prefix, savepath_prefix, load_data_dir, nz, model_str_list)

    shap_plot(prefix, savepath_prefix, nz, model_str_dict)
