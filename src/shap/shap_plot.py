import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import shap
import pandas as pd

import src.shap.utils




def global_plot(key, classifier_key, model_str, model_pos):
    clf = pickle.load(open(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pickle'), "rb"))
    # print(clf.classes_)

    with open(os.path.join('src/shap/save-shap-values', key, classifier_key + '-shap.sav'), 'rb') as f:
        shap_values = pickle.load(f)
    print(shap_values.shape)

    fig = plt.figure()

    shap.plots.beeswarm(shap_values[:, :, model_pos], max_display=20, show=False)
    # shap.plots.beeswarm(shap_values, max_display=20, show=False)

    plt.tight_layout()
    plt.savefig(os.path.join('src/shap/plot', key, classifier_key + '-beeswarm-' + model_str.split('_')[0] + '.png'))
    plt.close()

    


def local_plot(key):
    pass




if __name__ == '__main__':
    global_plot('NIHAOrt_TNG', 'xgboost', 'AGNrt_2times', 0)