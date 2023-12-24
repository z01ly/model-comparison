import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rtdl_revisiting_models import FTTransformer

import src.classification.utils as utils
from src.classification.simple_nn import SimpleNN, ResNetNN



def train(key, classifier_key, X, y, input_size, output_size, batch_size, num_epochs, gpu_id, use_cuda=True):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for class_label, int_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"Class '{class_label}' is transformed to encoding integer: {int_label}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    train_dataset = utils.TrainValDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if classifier_key == 'nn':
        nn_clf = SimpleNN(input_size, output_size)
    elif classifier_key == 'resnet':
        nn_clf = ResNetNN(input_size, output_size)
    elif classifier_key == 'transformer':
        default_kwargs = FTTransformer.get_default_kwargs()
        nn_clf = FTTransformer(
            n_cont_features=input_size,
            cat_cardinalities=[],
            d_out=output_size,
            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            )

    if use_cuda:
        nn_clf = nn_clf.cuda(gpu_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_clf.parameters(), weight_decay=1e-4) 
    optimizer = torch.optim.AdamW(
    # Instead of model.parameters(),
    model.make_parameter_groups(),
    lr=1e-4,
    weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

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

        scheduler.step()

    torch.save(nn_clf.state_dict(), os.path.join('src/classification/save-model/', key, classifier_key + '-model.pt'))

    return scaler, train_losses, avg_train_losses



def test(scaler, key, model_names, classifier_key, sdss_test_data, input_size, output_size, batch_size, gpu_id, use_cuda=True):
    sdss_test_scaled = scaler.transform(sdss_test_data)
    sdss_test_tensor = torch.tensor(sdss_test_scaled, dtype=torch.float32)

    sdss_test_dataset = utils.TestDataset(sdss_test_tensor)
    sdss_test_loader = DataLoader(sdss_test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if classifier_key == 'nn':
        nn_clf = SimpleNN(input_size, output_size)
    elif classifier_key == 'resnet':
        nn_clf = ResNetNN(input_size, output_size)
    elif classifier_key == 'transformer':
        nn_clf = FTTransformer(
            n_cont_features=input_size,
            cat_cardinalities=[],
            d_out=output_size,
            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            )

    nn_clf.load_state_dict(torch.load(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pt')))
    if use_cuda:
        nn_clf = nn_clf.cuda(gpu_id)
    nn_clf.eval()

    prob_list = []

    with torch.no_grad():
        for i, inputs in enumerate(sdss_test_loader):
            if use_cuda:
                inputs = inputs.cuda(gpu_id)
            outputs = nn_clf(inputs)

            probabilities = F.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()
            prob_list.append(probabilities)

    sdss_probs = np.concatenate(prob_list, axis=0)
    print(sdss_probs.shape)

    sdss_probs_df = pd.DataFrame(sdss_probs, columns=model_names)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=sdss_probs_df, palette="Set3")
    plt.xlabel("Models")
    plt.ylabel("Probability")
    plt.title("Violin Plot of Predicted Probabilities")
    plt.savefig(os.path.join('src/classification/violin-plot/', key, classifier_key + '-violin.png'))
    plt.close()



if __name__ == "__main__":
    input_size = 32
    output_size = 6
    batch_size = 8
    num_epochs = 30
    gpu_id = 2

    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    X, y = utils.load_data_train(model_names)

    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    print(sdss_test_data.shape)

    scaler, train_losses, avg_train_losses = train('NIHAOrt_TNG', 'transformer', X, y, 
        input_size, output_size, batch_size, num_epochs, gpu_id, use_cuda=True)
    test(scaler, 'NIHAOrt_TNG', [s.split('_')[0] for s in model_names], 'transformer', sdss_test_data, 
        input_size, output_size, batch_size, gpu_id, use_cuda=True)