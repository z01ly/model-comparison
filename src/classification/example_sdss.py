import numpy as np
import matplotlib

from src.data.utils import filter_df
matplotlib.use('Agg')
import pickle
import os

from sklearn.preprocessing import LabelEncoder

import src.data.utils


def map_model_idx(model_str_list, print_key=False):
    label_binarizer = LabelEncoder().fit(model_str_list)
    model_idx_dict = {}
    for class_label, int_label in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        model_idx_dict[class_label] = int_label
        if print_key:
            print(f"{class_label}: {model_idx_dict[class_label]}")

    return model_idx_dict


# plot some example images that have high probability and some that have low probability
def main(low, high, model_str, model_str_list, save_dir, classifier_key, sdss_test_df, nz):
    model_idx_dict = map_model_idx(model_str_list)
    
    clf = pickle.load(open(os.path.join(save_dir, 'save-model', classifier_key + '-model.pickle'), "rb"))
    with open(os.path.join(save_dir, 'save-scaler', classifier_key + '-scaler.pickle'), 'rb') as f:
        scaler = pickle.load(f)
    
    # scaling
    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()
    sdss_test_scaled = scaler.transform(sdss_test_data)
    sdss_pred_prob = clf.predict_proba(sdss_test_scaled)

    current_col = sdss_pred_prob[:, model_idx_dict[model_str]]

    low_idx = np.where(current_col < low)[0]
    # low_arr = current_col[current_col < low]
    print(len(low_idx))

    high_idx = np.where(current_col > high)[0]
    # high_arr= current_col[current_col > high]
    print(len(high_idx))

    df_dir = os.path.join(save_dir, 'example-sdss', model_str)
    filter_df(sdss_test_df, low_idx, df_dir, 'low')
    destination_dir = os.path.join(df_dir, 'low')
    os.makedirs(destination_dir, exist_ok=True)
    src.data.utils.copy_df_path_images(df_dir, destination_dir, 'low')

    filter_df(sdss_test_df, high_idx, df_dir, 'high')
    destination_dir = os.path.join(df_dir, 'high')
    os.makedirs(destination_dir, exist_ok=True)
    src.data.utils.copy_df_path_images(df_dir, destination_dir, 'high')

