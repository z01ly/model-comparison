import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier, StackingClassifier

import src.classification.utils as utils
import src.classification.bayesflow_calibration as bayesflow_calibration



def main(save_dir, model_names, X, y, classifier_key, cuda_num, max_iter=300):
    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    # for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
    #     print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    clf_MLP = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', \
            alpha=0.01, learning_rate='adaptive', max_iter=max_iter, random_state=42)

    clf_RF = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')

    clf_XGB = XGBClassifier(objective='multi:softmax', tree_method='hist', device='cuda:'+cuda_num, verbosity=0) 

    
    if classifier_key == 'stacking-MLP-RF-XGB':
        meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
        clf = StackingClassifier(estimators=[('MLP', clf_MLP), ('RF', clf_RF), ('XGB', clf_XGB)], final_estimator=meta_classifier)
    elif classifier_key == 'voting-MLP-RF-XGB':
        clf = VotingClassifier(estimators=[('MLP', clf_MLP), ('RF', clf_RF), ('XGB', clf_XGB)], voting='soft')
    elif classifier_key == 'single-MLP':
        clf = clf_MLP


    scaler = StandardScaler()

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

        # scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf.fit(X_train_scaled, y_train_onehot)

        y_pred_onehot = clf.predict(X_test_scaled)
        y_pred = label_binarizer.inverse_transform(y_pred_onehot)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        confusion_matrices.append(cm)

        fold_probabilities = clf.predict_proba(X_test_scaled)
        all_probabilities.append(fold_probabilities)
        all_true_labels.append(y_test)


    probabilities = np.concatenate(all_probabilities, axis=0)
    true_labels = np.concatenate(all_true_labels, axis=0)

    bayesflow_onehot = LabelBinarizer()
    true_labels_onehot = bayesflow_onehot.fit_transform(true_labels)
    cal_curves = bayesflow_calibration.plot_calibration_curves(true_models=true_labels_onehot,
                                                               pred_models=probabilities,
                                                               model_names=model_names,
                                                               label_fontsize=18,
                                                               legend_fontsize=18,
                                                               title_fontsize=20,
                                                               tick_fontsize=16)
    plt.savefig(os.path.join(save_dir, 'calibration-curve', classifier_key + '-cc.png'))
    plt.close()

    # average_cm = np.mean(confusion_matrices, axis=0)
    sum_cm = np.sum(confusion_matrices, axis=0) / n_repeats / n_splits

    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_binarizer.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')

    plt.title(classifier_key + ' Confusion Matrix', fontsize=20)

    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=12)
    for text in disp.text_.ravel():
        text.set_fontsize(14)

    plt.savefig(os.path.join(save_dir, 'confusion-matrix', classifier_key + '-cm.png'))
    plt.close()





if __name__ == "__main__":
    # MLP candidate_architectures: [(64, 32), (64, 64), (128, 64), (128, 128), (128, 64, 32), (32, 32, 32), (64, 64, 64)]
    # utils.pre_makedirs()

    # keep list order
    # nihao_list = ['AGN', 'NOAGN', 'UHD_2times', 'mockobs_0915', 'n80_2times']
    # illustris_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'illustris-1_snapnum_135']
    # compare_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'mockobs_0915_2times']
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    classifier_keys = ['single-MLP', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    for classifier_key in classifier_keys:
        main('NIHAOrt_TNG', [s.split('_')[0] for s in model_names], X, y, classifier_key)

