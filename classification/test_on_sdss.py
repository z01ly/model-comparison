import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pickle

import utils


def train(classifier_key):
    X, y = utils.load_data_train()

    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    if classifier_key == 'random-forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    elif classifier_key == 'xgboost':
        clf = XGBClassifier(objective='multi:softmax', tree_method='gpu_hist', gpu_id=1)

    clf.fit(X, y_onehot)

    pickle.dump(clf, open('./save-model/' + classifier_key + '-model.pickle', 'wb'))

    

def test(classifier_key):
    # current: selected sdss
    sdss_test_data = np.load('../infoVAE/test_results/latent/selected_sdss_test.npy')
    print(sdss_test_data.shape)

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
    train('random-forest')
    train('xgboost') 

    test('random-forest')
    test('xgboost')