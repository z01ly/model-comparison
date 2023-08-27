import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from sklearn.neural_network import MLPClassifier

import main


def cross_val(classifier_key):
    X, y = main.load_data_train()
    
    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)

    scaler = StandardScaler()

    n_splits = 5
    n_repeats = 3
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

    plt.figure(figsize=(10, 10))
    markers = ['o', 'v', 's', 'p']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for model_idx, model_label in enumerate(label_binarizer.inverse_transform(clf.classes_)):
        prob_true, prob_pred = calibration_curve(true_labels == model_label, probabilities[:, model_idx], n_bins=10)
        plt.plot(prob_pred, prob_true, marker=markers[model_idx], linestyle=linestyles[model_idx], label=f'{model_label}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Empirical Probability')
    plt.title('Calibration Curve: ' + classifier_key)
    plt.legend()
    plt.savefig('./calibration-curve/' + classifier_key + '-cc.png')
    plt.close()


    # average_cm = np.mean(confusion_matrices, axis=0)
    sum_cm = np.sum(confusion_matrices, axis=0) / n_repeats / n_splits

    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_binarizer.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')

    plt.title(classifier_key + ' Confusion Matrix')
    plt.savefig('./confusion-matrix/' + classifier_key + '-cm.png')
    plt.close()





if __name__ == "__main__":
    cross_val('MLP')