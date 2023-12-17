import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import src.classification.bayesflow_calibration as bayesflow_calibration
import src.classification.utils as utils
from src.classification.simple_nn import SimpleNN


# kfold ref: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
def main(key, model_names, X, y, classifier_key,
        input_size, hidden_size1, hidden_size2, output_size, dropout_rate, batch_size,
        num_epochs, gpu_id, use_cuda=True):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    for class_label, int_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"Class '{class_label}' is transformed to encoding integer: {int_label}")

    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    n_splits = 5
    n_repeats = 2
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

    confusion_matrices = []
    all_probs_list = []
    all_true_labels_list = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

        train_dataset = utils.CustomDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        test_dataset = utils.CustomDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        nn_clf = SimpleNN(input_size, hidden_size1, hidden_size2, output_size, dropout_rate)
        if use_cuda:
            nn_clf = nn_clf.cuda(gpu_id)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_clf.parameters())

        train_losses = []
        avg_train_losses = []

        for epoch in range(num_epochs):
            nn_clf.train()
            sum_train_loss_epoch = 0
            itr = 0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                if use_cuda:
                    inputs = inputs.cuda(gpu_id)
                outputs = nn_clf(inputs)

                if use_cuda:
                    labels = labels.cuda(gpu_id)
                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()
            
                train_losses.append(loss.item())
                sum_train_loss_epoch += loss.item()
                itr += 1

            train_avg = sum_train_loss_epoch / itr # average train loss per epoch
            avg_train_losses.append(train_avg)


        fold_labels_list = [] # true labels
        fold_predicted_list = []
        fold_prob_list = []

        nn_clf.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if use_cuda:
                    inputs = inputs.cuda(gpu_id)
                outputs = nn_clf(inputs)

                probabilities = F.softmax(outputs, dim=1)
                probabilities = probabilities.cpu().numpy()
                fold_prob_list.append(probabilities)

                _, predicted = torch.max(outputs, 1) # max in each row
                predicted = predicted.cpu().numpy()
                fold_predicted_list.append(label_encoder.inverse_transform(predicted))

                labels = labels.numpy()
                fold_labels_list.append(label_encoder.inverse_transform(labels))

        fold_labels = np.concatenate(fold_labels_list, axis=0)
        fold_predicteds = np.concatenate(fold_predicted_list, axis=0)
        fold_probs = np.concatenate(fold_prob_list, axis=0)

        cm = confusion_matrix(fold_labels, fold_predicteds, normalize='true')
        confusion_matrices.append(cm)

        all_probs_list.append(fold_probs)
        all_true_labels_list.append(fold_labels)

    all_probabilities = np.concatenate(all_probs_list, axis=0)
    all_true_labels = np.concatenate(all_true_labels_list, axis=0)

    bayesflow_onehot = LabelBinarizer()
    all_true_labels_onehot = bayesflow_onehot.fit_transform(all_true_labels)
    cal_curves = bayesflow_calibration.plot_calibration_curves(all_true_labels_onehot, all_probabilities, model_names)
    plt.savefig(os.path.join('src/classification/calibration-curve', key, classifier_key + '-cc.png'))
    plt.close()

    sum_cm = np.sum(confusion_matrices, axis=0) / n_repeats / n_splits
    disp = ConfusionMatrixDisplay(confusion_matrix=sum_cm, display_labels=label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.2f')

    plt.title(classifier_key + ' Confusion Matrix')
    plt.savefig(os.path.join('src/classification/confusion-matrix', key, classifier_key + '-cm.png'))
    plt.close()





if __name__ == "__main__":
    # y_onehot = F.one_hot(torch.tensor(y_integer), num_classes=len(model_names))
    input_size = 32
    hidden_size1 = 64
    hidden_size2 = 128
    output_size = 6
    dropout_rate = 0.5
    batch_size = 128
    num_epochs= 20
    gpu_id = 2

    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    main('NIHAOrt_TNG', model_names, X, y, 'nn', 
        input_size, hidden_size1, hidden_size2, output_size, dropout_rate, batch_size,
        num_epochs, gpu_id, use_cuda=True)


    