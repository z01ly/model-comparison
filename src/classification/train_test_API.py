import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier, StackingClassifier

import src.classification.utils as utils




def train(save_dir, key, classifier_key, X, y, max_iter=400):
    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    # for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
    #     print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")


    clf_MLP = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', \
            alpha=0.01, learning_rate='adaptive', max_iter=max_iter, random_state=42, early_stopping=True)

    clf_RF = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')

    clf_XGB = XGBClassifier(objective='multi:softmax', tree_method='hist', device='cuda:0', verbosity=0)


    if classifier_key == 'stacking-MLP-RF-XGB':
        meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
        clf = StackingClassifier(estimators=[('MLP', clf_MLP), ('RF', clf_RF), ('XGB', clf_XGB)], final_estimator=meta_classifier)
    elif classifier_key == 'voting-MLP-RF-XGB':
        clf = VotingClassifier(estimators=[('MLP', clf_MLP), ('RF', clf_RF), ('XGB', clf_XGB)], voting='soft')
    elif classifier_key == 'single-MLP':
        clf = clf_MLP

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf.fit(X_scaled, y_onehot)

    pickle.dump(clf, open(os.path.join(save_dir, 'save-model', key, classifier_key + '-model.pickle'), 'wb'))

    return scaler



def test(save_dir, scaler, key, model_names, classifier_key, sdss_test_data):
    clf = pickle.load(open(os.path.join(save_dir, 'save-model', key, classifier_key + '-model.pickle'), "rb"))

    # label_binarizer = LabelEncoder().fit(model_names)
    # for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
    #     print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    # scaling
    sdss_test_scaled = scaler.transform(sdss_test_data)

    sdss_pred_prob = clf.predict_proba(sdss_test_scaled)
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
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    classifier_keys = ['tabnet'] # 'single-MLP', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB'
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    print(sdss_test_data.shape)
    for classifier_key in classifier_keys:
        scaler = train('NIHAOrt_TNG', classifier_key, X, y)
        test(scaler, 'NIHAOrt_TNG', [s.split('_')[0] for s in model_names], classifier_key, sdss_test_data)