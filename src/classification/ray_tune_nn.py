# This file is adapted from: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
# and https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import tempfile
from filelock import FileLock
from typing import Dict
import ray
from ray import train, tune
from ray.train import Result
from ray.tune import ResultGrid
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import src.classification.utils as utils
from src.classification.simple_nn import SimpleNN



def train_tabular(config):
    # if classifier_key == 'nn':
    input_size = 32
    output_size = 6
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    net = SimpleNN(input_size, output_size, config["r1"], config["r2"], config["r3"], config["r4"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler_gamma"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    X, y = utils.load_data(model_names, switch='train')
    X_tensor, y_tensor = utils.pre_nn_data(X, y)
    trainset = utils.TrainValDataset(X_tensor, y_tensor)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
    )

    for epoch in range(20):  # loop over the dataset multiple times
        # training
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # scheduler
        scheduler.step()

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")



def test_best_model(best_result, model_names, input_size, output_size):
    best_trained_model = SimpleNN(input_size, output_size, best_result.config["r1"], best_result.config["r2"], best_result.config["r3"], best_result.config["r4"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    # print(checkpoint_path)

    model_state, optimizer_state = torch.load(checkpoint_path)

    # in compatible with net = nn.DataParallel(net)
    model_state = {key.replace('module.', ''): value for key, value in model_state.items()}

    best_trained_model.load_state_dict(model_state)

    # get scaler from training data
    X_train, y_train = utils.load_data(model_names, switch='train')
    _, _, _, scaler = utils.pre_nn_data(X_train, y_train)
    X, y = utils.load_data(model_names, switch='test')
    X_tensor, y_tensor, _, _ = utils.pre_nn_data(X, y, scaler)

    testset = utils.TrainValDataset(X_tensor, y_tensor)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Best trial test set accuracy: {}".format(correct / total))



def main(model_names, input_size, output_size, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "r1": tune.sample_from(lambda _: 2 ** np.random.randint(8, 9)),
        "r2": tune.sample_from(lambda _: 2 ** np.random.randint(7, 8)),
        "r3": tune.sample_from(lambda _: 2 ** np.random.randint(6, 7)),
        "r4": tune.sample_from(lambda _: 2 ** np.random.randint(5, 6)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "scheduler_gamma": tune.choice([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
    }
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            # train_tabular have more parameters? 
            tune.with_parameters(train_tabular),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result, model_names, input_size, output_size)



def read_results():
    storage_path = "/export/home/lzhou/ray_results"
    exp_name = "train_tabular_2024-01-09_10-13-08"
    experiment_path = os.path.join(storage_path, exp_name)
    print(f"...Loading results from {experiment_path}")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_tabular)
    result_grid = restored_tuner.get_results()

    if result_grid.errors:
        print("One of the trials failed!")
    else:
        print("No errors!")

    num_results = len(result_grid)
    print("Number of results:", num_results)

    # Iterate over results
    for i, result in enumerate(result_grid):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a mean accuracy metric of:",
            result.metrics['accuracy']
        )

    best_result: Result = result_grid.get_best_result()
    test_best_model(best_result, model_names, input_size, output_size)
    # print(best_result.config)
    # print(best_result.path)
    # print(best_result.metrics['accuracy'])




if __name__ == "__main__":
    # key = 'NIHAOrt_TNG'
    # classifier_key = 'nn'
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    input_size = 32
    output_size = 6
    # gpu_id = 2

    main(model_names, input_size, output_size, num_samples=10, max_num_epochs=10, gpus_per_trial=2)
    # read_results()