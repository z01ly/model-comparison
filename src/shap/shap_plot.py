import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

import src.shap.utils
import shap



def global_plot(key, classifier_key):
    clf = pickle.load(open(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pickle'), "rb"))
    # print(clf.classes_)

    with open(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.sav'), 'rb') as f:
        shap_values = pickle.load(f)
    print(shap_values.shape)

    fig = plt.figure()

    # shap.plots.beeswarm(shap_values[:, :, 0], max_display=20, show=False)
    shap.plots.beeswarm(shap_values, max_display=20, show=False)

    plt.tight_layout()
    plt.savefig(os.path.join('src/shap/plot', key, classifier_key, 'global', 'beeswarm.png'))
    plt.close()

    



def local_plot(key):
    pass




if __name__ == '__main__':
    # src.shap.utils.mkdirs()

    # global_plot('compare', 'xgboost')
    # global_plot('compare', 'random-forest')
    global_plot('compare', 'single-MLP')
    global_plot('compare', 'voting-MLP-RF-XGB')
    global_plot('compare', 'stacking-MLP-RF-XGB')
