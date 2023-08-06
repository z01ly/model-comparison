import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

from xgboost import XGBClassifier


def classification(encoder_key, classifier_key):
    AGN_data = np.load('../infoVAE/test_results/latent/AGN.npy')
    AGN_labels = np.full(AGN_data.shape[0], 'AGN') 

    NOAGN_data = np.load('../infoVAE/test_results/latent/NOAGN.npy')
    NOAGN_labels = np.full(NOAGN_data.shape[0], 'NOAGN') 

    n80_data = np.load('../infoVAE/test_results/latent/n80.npy')
    n80_labels = np.full(n80_data.shape[0], 'n80') 

    UHD_data = np.load('../infoVAE/test_results/latent/UHD.npy')
    UHD_labels = np.full(UHD_data.shape[0], 'UHD') 

    X = np.concatenate((AGN_data, NOAGN_data, n80_data, UHD_data), axis=0)
    y = np.concatenate((AGN_labels, NOAGN_labels, n80_labels, UHD_labels), axis=0)

    random_indices = np.random.permutation(len(X)) # data shuffle
    X = X[random_indices]
    y = y[random_indices]


    if encoder_key == 'one-hot':
        label_binarizer = LabelBinarizer()
        y_onehot = label_binarizer.fit_transform(y)
    elif encoder_key == 'integer':
        label_binarizer = LabelEncoder()
        y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")


    if classifier_key == 'random-forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    elif classifier_key == 'balanced-random-forest':
        clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, \
                sampling_strategy='all', replacement=True)
    elif classifier_key == 'xgboost':
        clf = XGBClassifier()
    elif classifier_key == 'logistic-regression':
        clf = LogisticRegression(multi_class='multinomial', class_weight='balanced', max_iter=500, solver='lbfgs')
    elif classifier_key == 'gradient-boosting':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif classifier_key == 'svc':
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=42, decision_function_shape='ovr')
    elif classifier_key == 'knn':
        clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='kd_tree')
    elif classifier_key == 'naive-bayes':
        clf = BernoulliNB()

    

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    confusion_matrices = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train_onehot = label_binarizer.transform(y_train)
        y_test_onehot = label_binarizer.transform(y_test)

        clf.fit(X_train, y_train_onehot)

        y_pred_onehot = clf.predict(X_test)

        y_pred = label_binarizer.inverse_transform(y_pred_onehot)
        
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)

    # average_cm = np.mean(confusion_matrices, axis=0)
    sum_cm = np.sum(confusion_matrices, axis=0)

    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_binarizer.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')

    plt.title(classifier_key + ' Confusion Matrix')
    plt.savefig('./confusion-matrix/' + classifier_key + '-cm.png')




if __name__ == "__main__":
    # imbalanced data
    classification('integer', 'naive-bayes')
    