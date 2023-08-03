import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier


def classification(key):
    AGN_data = np.load('../infoVAE/test_results/latent/AGN.npy')
    AGN_labels = np.full(AGN_data.shape[0], 'AGN') 

    NOAGN_data = np.load('../infoVAE/test_results/latent/NOAGN.npy')
    NOAGN_labels = np.full(NOAGN_data.shape[0], 'NOAGN') 

    UHD_data = np.load('../infoVAE/test_results/latent/UHD.npy')
    UHD_labels = np.full(UHD_data.shape[0], 'UHD') 

    n80_data = np.load('../infoVAE/test_results/latent/n80.npy')
    n80_labels = np.full(n80_data.shape[0], 'n80') 

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
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    confusion_matrices = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train_onehot = label_binarizer.transform(y_train)
        y_test_onehot = label_binarizer.transform(y_test)

        rf_classifier.fit(X_train, y_train_onehot)

        y_pred_onehot = rf_classifier.predict(X_test)

        y_pred = label_binarizer.inverse_transform(y_pred_onehot)
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

    # average_cm = np.mean(confusion_matrices, axis=0)
    sum_cm = np.sum(confusion_matrices, axis=0)

    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_binarizer.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('./random_forest_cm.png')

"""
    plt.imshow(sum_cm, cmap='Blues')

    num_classes = len(label_binarizer.classes_)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, label_binarizer.classes_)
    plt.yticks(tick_marks, label_binarizer.classes_)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(sum_cm[i, j]), horizontalalignment="center", color="black")

    plt.colorbar()
    plt.savefig('./random_forest_cm.png')
"""




if __name__ == "__main__":
    classification()