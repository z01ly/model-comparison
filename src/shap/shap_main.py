import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

import src.shap.utils
from src.classification.utils import load_data_train

import shap


def save_shap_values(background, test_data, key, classifier_key):
    print(f"Background data shape: {background.shape}")
    print(f"Test data shape: {test_data.shape}")

    clf = pickle.load(open(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pickle'), "rb"))

    explainer = shap.TreeExplainer(clf, background) # training data of clf as background data
    shap_values = explainer(test_data) # how to save
    print('\n')

    # np.save(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.npy'), shap_values)
    # with open(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.sav'), "wb") as f_stream:
    #     shap_values.save(f_stream)

    # shap.plots.beeswarm(shap_values)

    print(type(shap_values))
    # shap.summary_plot(shap_values[:, :, 0])
    shap.plots.beeswarm(shap_values[:, :, 0])
    plt.savefig(os.path.join('src/shap/plot', key, classifier_key, 'global', 'beeswarm.png'))
    plt.close()




if __name__ == '__main__':
    # src.shap.utils.mkdirs()
    compare_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'mockobs_0915_2times']
    X, y = load_data_train(compare_list)

    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')

    save_shap_values(X, sdss_test_data[0: 1000], 'compare', 'xgboost')
    
