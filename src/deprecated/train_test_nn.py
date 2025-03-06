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

from pytorch_tabnet.tab_network import TabNet



def train(key, classifier_key, model_names, input_size, output_size, batch_size, num_epochs, gpu_id, use_cuda=True):
    X, y = utils.load_data(model_names, switch='train')
    X_tensor, y_tensor, label_encoder, scaler = utils.pre_nn_data(X, y)

    train_dataset = utils.TrainValDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if classifier_key == 'nn':
        nn_clf = SimpleNN(input_size, output_size)
    elif classifier_key == 'resnet':
        nn_clf = ResNetNN(input_size, output_size)
    elif classifier_key == 'fttransformer':
        default_kwargs = FTTransformer.get_default_kwargs()
        nn_clf = FTTransformer(
            n_cont_features=input_size,
            cat_cardinalities=[],
            d_out=output_size,
            **default_kwargs,
            )
    elif classifier_key == 'tabnet':
        eye_matrix = torch.eye(input_size)
        if use_cuda:
            eye_matrix = eye_matrix.cuda(gpu_id)
        nn_clf = TabNet(input_size, output_size, group_attention_matrix=eye_matrix)

    if use_cuda:
        nn_clf = nn_clf.cuda(gpu_id)

    criterion = nn.CrossEntropyLoss()
    # if classifier_key == 'fttransformer':
    #     optimizer = nn_clf.make_default_optimizer()
    # else:
    #     optimizer = torch.optim.Adam(nn_clf.parameters(), weight_decay=1e-4)
    optimizer = torch.optim.Adam(nn_clf.parameters(), lr=0.000373, weight_decay=1e-4)
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
               
            # tabnet
            # outputs, _ = nn_clf(inputs)

            # ft transformer
            outputs = nn_clf(inputs, None)

            # general 
            # outputs = nn_clf(inputs)

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
        print(f"epoch: {epoch}, average loss: {train_avg}")

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
    elif classifier_key == 'fttransformer':
        nn_clf = FTTransformer(
            n_cont_features=input_size,
            cat_cardinalities=[],
            d_out=output_size,
            **FTTransformer.get_default_kwargs(),
            )
    elif classifier_key == 'tabnet':
        eye_matrix = torch.eye(input_size)
        if use_cuda:
            eye_matrix = eye_matrix.cuda(gpu_id)
        nn_clf = TabNet(input_size, output_size, group_attention_matrix=eye_matrix)

    nn_clf.load_state_dict(torch.load(os.path.join('src/classification/save-model/', key, classifier_key + '-model.pt')))
    if use_cuda:
        nn_clf = nn_clf.cuda(gpu_id)
    nn_clf.eval()

    prob_list = []

    with torch.no_grad():
        for i, inputs in enumerate(sdss_test_loader):
            if use_cuda:
                inputs = inputs.cuda(gpu_id)

            # tabnet
            # outputs, _ = nn_clf(inputs)

            # ft transformer
            outputs = nn_clf(inputs, None)

            # general 
            # outputs = nn_clf(inputs)

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
    # batch_size = 8
    # num_epochs = 30
    batch_size = 32 # 128
    num_epochs = 100 # 80
    gpu_id = 2

    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']

    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    print(sdss_test_data.shape)

    key = 'NIHAOrt_TNG'
    classifier_key = 'fttransformer'
    
    scaler, train_losses, avg_train_losses = train(key, classifier_key, model_names, 
        input_size, output_size, batch_size, num_epochs, gpu_id, use_cuda=True)
    test(scaler, key, [s.split('_')[0] for s in model_names], classifier_key, sdss_test_data, 
        input_size, output_size, batch_size, gpu_id, use_cuda=True)

    os.makedirs(os.path.join('src/classification/simple-nn', key, 'train-test'), exist_ok=True)
    utils.train_loss_plot(train_losses, 'iterations', os.path.join('src/classification/simple-nn', key, 'train-test', classifier_key + '_itr_loss.png'))
    utils.train_loss_plot(avg_train_losses, 'epochs', os.path.join('src/classification/simple-nn', key, 'train-test', classifier_key + '_avg_loss.png'))
