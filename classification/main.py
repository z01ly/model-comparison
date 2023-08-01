import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def classification():
    AGN_data = np.load('../infoVAE/test_results/latent/AGN.npy')
    AGN_labels = np.zeros(AGN_data.shape[0]) # 0 -> [1, 0, 0, 0]

    NOAGN_data = np.load('../infoVAE/test_results/latent/NOAGN.npy')
    NOAGN_labels = np.ones(NOAGN_data.shape[0]) # 1 -> [0, 1, 0, 0]

    n80_data = np.load('../infoVAE/test_results/latent/n80.npy')
    n80_labels = np.full(n80_data.shape[0], 2) # 2 -> [0, 0, 1, 0]

    UHD_data = np.load('../infoVAE/test_results/latent/UHD.npy')
    UHD_labels = np.full(UHD_data.shape[0], 3) # 3 -> [0, 0, 0, 1]

    X = np.concatenate((AGN_data, NOAGN_data, n80_data, UHD_data), axis=0)
    y = np.concatenate((AGN_labels, NOAGN_labels, n80_labels, UHD_labels), axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]

    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to one-hot encoding vector: {onehot_vector}")

    # later maybe: BalancedRandomForestClassifier from imbalanced-learn lib
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced_subsample') # class weight for imbalanced data

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    kf = KFold(n_splits=5, shuffle=False)

    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train_onehot = label_binarizer.transform(y_train)
        y_test_onehot = label_binarizer.transform(y_test)

        rf_classifier.fit(X_train, y_train_onehot)

        y_pred_onehot = rf_classifier.predict(X_test)

        y_pred = label_binarizer.inverse_transform(y_pred_onehot)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    average_accuracy = np.mean(accuracies)
    print("Average Accuracy:", average_accuracy)





if __name__ == "__main__":
    classification()