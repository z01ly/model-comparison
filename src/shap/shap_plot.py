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

    with open(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.sav'), 'rb') as f:
        shap_values = pickle.load(f)

    fig = plt.figure()

    # shap.plots.beeswarm(shap_values[:, :, 0], max_display=20, show=False)
    # shap.summary_plot(shap_values, class_names=clf.classes_)
    shap.summary_plot(shap_values[:, :, 0], class_names=['TNG100-1', 'TNG50-1', 'mockobs'])

    plt.tight_layout()
    plt.savefig(os.path.join('src/shap/plot', key, classifier_key, 'global', 'summary.png'))
    plt.close()



def local_plot(key):
    pass




if __name__ == '__main__':
    # utils.mkdirs()
    global_plot('compare', 'xgboost')

