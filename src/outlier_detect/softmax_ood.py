import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

from scipy.stats import ttest_1samp, wilcoxon

from xgboost import XGBClassifier

import src.classification.utils


def softmax_ood(model_names):
    clf = XGBClassifier(objective='multi:softmax', tree_method='hist', device='cuda:1')

    X_train, y_train = src.classification.utils.load_data(model_names, switch='train')
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    X_val, y_val = src.classification.utils.load_data(model_names, switch='test')
    y_val_encoded = label_encoder.transform(y_val)

    clf.fit(X_train, y_train_encoded)

    y_val_pred = clf.predict(X_val)
    softmax_scores_val = clf.predict_proba(X_val)
    confidences_val = np.max(softmax_scores_val, axis=1)

    """
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for threshold in thresholds:
        mask = (confidences_val > threshold).astype(int)

        positive_indices = np.where(mask == 1)[0]

        y_val_subset = y_val_encoded[positive_indices]
        y_val_pred_subset = y_val_pred[positive_indices]

        f1 = f1_score(y_val_subset, y_val_pred_subset, average='micro')
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    print("Best threshold for maximum f1 score:", best_threshold)
    # f1_scores = [f1_score(y_val_encoded, confidences_val > threshold, average='micro') for threshold in thresholds]
    # optimal_threshold = thresholds[np.argmax(f1_scores)]
    # print(optimal_threshold)
    """

    null_hypothesis_mean = 0.895
    # t_statistic, p_value = ttest_1samp(confidences_val, null_hypothesis_mean)
    statistic, p_value = wilcoxon(confidences_val - null_hypothesis_mean)
    print(f"p_value: {p_value}")

    """
    thresholds = np.linspace(0.6, 1, 100)
    for tr in thresholds:
        statistic, p_value = wilcoxon(confidences_val - tr)
        if p_value > 0.01:
            print(tr)
            print(p_value)
    """ 

    best_threshold = null_hypothesis_mean
    print("Best threshold: ", best_threshold)
    
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')

    softmax_scores = clf.predict_proba(sdss_test_data)

    confidences = np.max(softmax_scores, axis=1)
    # neg_confidences = -confidences
    plt.hist(confidences_val, bins=40, density=True, histtype='step', label="valset")
    plt.hist(confidences, bins=40, density=True, histtype='step', label="sdss testset")
    plt.axvline(x=null_hypothesis_mean, color='red', linestyle='--', label='threshold')
    plt.text(null_hypothesis_mean, plt.ylim()[0], f'{null_hypothesis_mean}', va='bottom', ha='center', color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.xlabel('confidences desity')
    plt.ylabel('number')
    plt.title('histogram of confidences')
    plt.legend()
    plt.savefig('src/distribution/softmax/confidence_hist.png', dpi=300)

    out_of_distribution_samples = np.where(confidences < best_threshold)[0]

    print(out_of_distribution_samples.shape[0] / confidences.shape[0])




if __name__ == "__main__":
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    softmax_ood(model_names)