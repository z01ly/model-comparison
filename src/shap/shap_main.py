import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split

import src.shap.utils
from src.classification.utils import load_data_train

import shap



def background_sample(X, y, percent=0.01):
    # stratify: to preserve class frequencies
    X_train, X_sampled, y_train, y_sampled = train_test_split(X, y, test_size=percent, stratify=y, random_state=42)

    return X_sampled



def save_shap_values(background, test_data, key, classifier_key, explainer_key):
    print(f"Background data shape: {background.shape}")
    print(f"Test data shape: {test_data.shape}")

    clf = pickle.load(open(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pickle'), "rb"))

    if explainer_key == 'Explainer':
        explainer = shap.Explainer(clf.predict, background)
    elif explainer_key == 'TreeExplainer':
        explainer = shap.TreeExplainer(clf, background) # training data of clf as background data
    elif explainer_key == 'KernelExplainer':
        explainer = shap.KernelExplainer(clf.predict, background)
    
    shap_values = explainer(test_data) # type(shap_values): <class 'shap._explanation.Explanation'>
    print('\n')

    with open(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.sav'), 'wb') as f:
        pickle.dump(shap_values, f)




if __name__ == '__main__':
    # src.shap.utils.mkdirs()

    compare_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'mockobs_0915_2times']
    X, y = load_data_train(compare_list)

    X_sampled = background_sample(X, y, percent=0.006)

    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')

    # save_shap_values(X, sdss_test_data, 'compare', 'xgboost', 'TreeExplainer')
    # save_shap_values(X, sdss_test_data, 'compare', 'random-forest', 'TreeExplainer')
    # save_shap_values(X, sdss_test_data, 'compare', 'single-MLP', 'Explainer')

    # divide by zero encountered in log
    # save_shap_values(X_sampled, sdss_test_data, 'compare', 'voting-MLP-RF-XGB', 'KernelExplainer') 
    save_shap_values(X_sampled, sdss_test_data[0:100], 'compare', 'voting-MLP-RF-XGB', 'Explainer') 

    # save_shap_values(X_sampled, sdss_test_data[0:100], 'compare', 'stacking-MLP-RF-XGB', 'Explainer')
    save_shap_values(X_sampled, sdss_test_data[0:100], 'compare', 'stacking-MLP-RF-XGB', 'KernelExplainer')
    
