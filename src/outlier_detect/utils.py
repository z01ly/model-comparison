import numpy as np
from PIL import Image
import os
import shutil
import random
import math
import pickle
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stack_mock_train_test(load_dir, model_str, z_dim, key):
    # key: 'all', 'inlier', 'outlier'
    # load dir example: 'src/results/latent-vectors'
    if key == 'all':
        train_dir = os.path.join(load_dir, 'train')
        test_dir = os.path.join(load_dir, 'test')
    elif key == 'inlier':
        train_dir = os.path.join(load_dir, 'train-inlier-original')
        test_dir = os.path.join(load_dir, 'test-inlier')
    elif key == 'outlier':
        train_dir = os.path.join(load_dir, 'train-outlier')
        test_dir = os.path.join(load_dir, 'test-outlier')
    
    train_df = pd.read_pickle(os.path.join(train_dir, model_str + '.pkl'))
    train_arr = train_df.iloc[:, 0:z_dim].to_numpy()

    test_df = pd.read_pickle(os.path.join(test_dir, model_str + '.pkl'))
    test_arr = test_df.iloc[:, 0:z_dim].to_numpy()

    stacked_data = np.vstack((train_arr, test_arr))
    # print(f"stacked data shape: {stacked_data.shape}")

    return stacked_data