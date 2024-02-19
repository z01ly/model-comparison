import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import pickle, os

import src.classification.utils as utils


def train(save_dir, key, classifier_key, X, y):
    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    if classifier_key == 'random-forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    elif classifier_key == 'xgboost':
        clf = XGBClassifier(objective='multi:softmax', tree_method='hist', device='cuda:0', verbosity=0)

    clf.fit(X, y_onehot)

    pickle.dump(clf, open(os.path.join(save_dir, 'save-model', key, classifier_key + '-model.pickle'), 'wb'))

    
    

def test(save_dir, key, model_names, classifier_key, sdss_test_data):
    clf = pickle.load(open(os.path.join(save_dir, 'save-model', key, classifier_key + '-model.pickle'), "rb"))

    label_binarizer = LabelEncoder().fit(model_names)
    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")
    
    """
    sdss_pred_onehot = clf.predict(sdss_test_data)
    sdss_pred = label_binarizer.inverse_transform(sdss_pred_onehot)

    total_elements = sdss_pred.shape[0]

    for target_class in ['AGN', 'NOAGN', 'UHD', 'mockobs_0915', 'n80']:
        class_count = np.count_nonzero(sdss_pred == target_class)
        percentage = (class_count / total_elements) * 100
        with open("test-output.txt", "a") as text_file:
            text_file.write(f"In {classifier_key} test, the percentage of occurrence of class {target_class}: {percentage:.2f}% \n")
    """

    sdss_pred_prob = clf.predict_proba(sdss_test_data)
    # model_names = ['AGN', 'NOAGN', 'UHD', 'mockobs_0915', 'n80']
    sdss_pred_prob_df = pd.DataFrame(sdss_pred_prob, columns=model_names)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=sdss_pred_prob_df, palette="Set3")
    plt.xlabel("Models")
    plt.ylabel("Probability")
    plt.title("Violin Plot of Predicted Probabilities")
    plt.savefig(os.path.join(save_dir, 'violin-plot', key, classifier_key + '-violin.png'))

    plt.close()
    



if __name__ == "__main__":
    # keep list order
    # illustris_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'illustris-1_snapnum_135']
    # compare_list = ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'mockobs_0915_2times']
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    train('NIHAOrt_TNG', 'random-forest', X, y)

    # current: all sdss test data
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    print(sdss_test_data.shape)
    test('NIHAOrt_TNG', [s.split('_')[0] for s in model_names], 'random-forest', sdss_test_data)