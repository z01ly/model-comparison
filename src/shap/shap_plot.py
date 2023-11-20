import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle

import src.shap.utils
import shap



def global_plot(key, classifier_key):
    shap_values = np.load(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.npy'))

    # shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)
    plt.savefig(os.path.join('src/shap/plot', key, classifier_key, 'global', 'beeswarm.png'))
    plt.close()


def local_plot(key):
    pass




if __name__ == '__main__':
    # utils.mkdirs()
    global_plot('compare', 'xgboost')

