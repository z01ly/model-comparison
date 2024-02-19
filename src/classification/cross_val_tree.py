import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

from sklearn.calibration import calibration_curve

import src.classification.utils as utils
import src.classification.bayesflow_calibration as bayesflow_calibration



def main(save_dir, key, model_names, X, y, encoder_key, classifier_key):
    if encoder_key == 'one-hot':
        label_binarizer = LabelBinarizer()
    elif encoder_key == 'integer':
        label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    # 'AGN' -> 0, 'NOAGN' -> 1, 'UHD' -> 2, 'mockobs_0915' -> 3, 'n80' -> 4
    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")


    if classifier_key == 'random-forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    elif classifier_key == 'xgboost':
        # https://github.com/dmlc/xgboost/issues/9791
        clf = XGBClassifier(objective='multi:softmax', tree_method='hist', device='cuda:0', verbosity=0)

    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    n_splits = 5
    n_repeats = 2
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    confusion_matrices = []

    all_probabilities = []
    all_true_labels = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train_onehot = label_binarizer.transform(y_train)
        y_test_onehot = label_binarizer.transform(y_test)
    
        clf.fit(X_train, y_train_onehot)
    
        y_pred_onehot = clf.predict(X_test)
        y_pred = label_binarizer.inverse_transform(y_pred_onehot)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        confusion_matrices.append(cm)

        fold_probabilities = clf.predict_proba(X_test)
        all_probabilities.append(fold_probabilities)
        all_true_labels.append(y_test)


    probabilities = np.concatenate(all_probabilities, axis=0)
    true_labels = np.concatenate(all_true_labels, axis=0)

    bayesflow_onehot = LabelBinarizer()
    true_labels_onehot = bayesflow_onehot.fit_transform(true_labels)
    # 'AGN' -> [1 0 0 0 0], 'NOAGN' -> [0 1 0 0 0], 'UHD' -> [0 0 1 0 0], 'mockobs_0915' -> [0 0 0 1 0], 'n80' -> [0 0 0 0 1]
    for class_label, onehot_vector in zip(bayesflow_onehot.classes_, bayesflow_onehot.transform(bayesflow_onehot.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")
    cal_curves = bayesflow_calibration.plot_calibration_curves(true_labels_onehot, probabilities, model_names)
    plt.savefig(os.path.join(save_dir, 'calibration-curve', key, classifier_key + '-cc.png'))
    plt.close()

    """
    plt.figure(figsize=(10, 10))
    markers = ['o', 'v', 's', 'p']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    # for model_idx, model_label in enumerate(['AGN', 'NOAGN', 'UHD', 'n80']):
    for model_idx, model_label in enumerate(label_binarizer.inverse_transform(clf.classes_)):
        prob_true, prob_pred = calibration_curve(true_labels == model_label, probabilities[:, model_idx], n_bins=10)
        plt.plot(prob_pred, prob_true, marker=markers[model_idx], linestyle=linestyles[model_idx], label=f'{model_label}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empirical Probability')
    plt.title('Calibration Curve: ' + classifier_key)
    plt.legend()
    plt.savefig('src/classification/calibration-curve/' + classifier_key + '-cc.png')
    plt.close()
    """


    # average_cm = np.mean(confusion_matrices, axis=0)
    sum_cm = np.sum(confusion_matrices, axis=0) / n_repeats / n_splits

    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_binarizer.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')

    plt.title(classifier_key + ' Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion-matrix', key, classifier_key + '-cm.png'))
    plt.close()

    



if __name__ == "__main__":
    # imbalanced data

    # keep list order
    # nihao_list = ['AGN', 'NOAGN', 'UHD_2times', 'mockobs_0915', 'n80_2times']
    # illustris_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'illustris-1_snapnum_135'] # keep this order
    # compare_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'mockobs_0915_2times']
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    main('NIHAOrt_TNG', [s.split('_')[0] for s in model_names], X, y, 'integer', 'random-forest')

    # 'balanced-random-forest', 'logistic-regression', 'gradient-boosting', 'svc', 'knn', 'naive-bayes'
    