import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import SVC

import pickle

import main

def train(classifier_key):
    X, y = main.load_data_train()

    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    if classifier_key == 'balanced-random-forest':
        clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, \
                sampling_strategy='all', replacement=True)
    elif classifier_key == 'svc':
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=42, decision_function_shape='ovr', probability=False)

    clf.fit(X, y_onehot)

    pickle.dump(clf, open('./save-model/' + classifier_key + '-model.pickle', 'wb'))

    

def test(classifier_key):
    sdss_test_data = np.load('../infoVAE/test_results/latent/sdss_test.npy')

    clf = pickle.load(open('./save-model/' + classifier_key + '-model.pickle', "rb"))

    label_binarizer = LabelEncoder().fit(['AGN', 'NOAGN', 'UHD', 'n80'])
    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")
    

    sdss_pred_onehot = clf.predict(sdss_test_data)
    sdss_pred = label_binarizer.inverse_transform(sdss_pred_onehot)

    total_elements = sdss_pred.shape[0]

    for target_class in ['AGN', 'NOAGN', 'UHD', 'n80']:
        class_count = np.count_nonzero(sdss_pred == target_class)
        percentage = (class_count / total_elements) * 100
        with open("test-output.txt", "a") as text_file:
            text_file.write(f"In {classifier_key} test, the percentage of occurrence of class {target_class}: {percentage:.2f}% \n")

    



if __name__ == "__main__":
    # train('balanced-random-forest')
    # train('svc')

    test('balanced-random-forest')
    test('svc')