import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from sklearn.neural_network import MLPClassifier

import pickle

import main



def cross_val(classifier_key, hidden_layer, max_iter=300):
    X, y = main.load_data_train()
    
    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='relu', solver='adam', \
        alpha=0.01, learning_rate='adaptive', max_iter=max_iter, random_state=42)

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



def train(classifier_key, hidden_layer, max_iter=300):
    X, y = main.load_data_train()

    label_binarizer = LabelEncoder()
    y_onehot = label_binarizer.fit_transform(y)

    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    clf = MLPClassifier(hidden_layer_sizes=hidden_layer, activation='relu', solver='adam', \
        alpha=0.01, learning_rate='adaptive', max_iter=max_iter, random_state=42, early_stopping=True)

    scaler = StandardScaler()

    # scaling
    X_scaled = scaler.fit_transform(X)

    clf.fit(X_scaled, y_onehot)

    pickle.dump(clf, open('./save-model/' + classifier_key + '-model.pickle', 'wb'))

    return scaler



def test(scaler, classifier_key):
    sdss_test_data = np.load('../infoVAE/test_results/latent/sdss_test.npy')

    clf = pickle.load(open('./save-model/' + classifier_key + '-model.pickle', "rb"))

    label_binarizer = LabelEncoder().fit(['AGN', 'NOAGN', 'UHD', 'n80'])
    for class_label, onehot_vector in zip(label_binarizer.classes_, label_binarizer.transform(label_binarizer.classes_)):
        print(f"Class '{class_label}' is transformed to encoding vector: {onehot_vector}")

    # scaling
    sdss_test_scaled = scaler.transform(sdss_test_data)

    sdss_pred_onehot = clf.predict(sdss_test_scaled)
    sdss_pred = label_binarizer.inverse_transform(sdss_pred_onehot)

    total_elements = sdss_pred.shape[0]
    
    for target_class in ['AGN', 'NOAGN', 'UHD', 'n80']:
        class_count = np.count_nonzero(sdss_pred == target_class)
        percentage = (class_count / total_elements) * 100
        with open("test-output.txt", "a") as text_file:
            text_file.write(f"In {classifier_key} test, the percentage of occurrence of class {target_class}: {percentage:.2f}% \n")






if __name__ == "__main__":
    # candidate_architectures = [(64, 32), (64, 64), (128, 64), (128, 128), (128, 64, 32), (32, 32, 32), (64, 64, 64)]
    candidate_architectures = [(128, 64)]
    
    for i in range(len(candidate_architectures)):
        classifier_key = f'MLP-v{i}'
        hidden_layer = candidate_architectures[i]

        cross_val(classifier_key, hidden_layer)
        scaler = train(classifier_key, hidden_layer)
        test(scaler, classifier_key)