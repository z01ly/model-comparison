import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scipy.special import softmax


def copy_previous_results():
    destination_dir = 'new'
    os.makedirs(destination_dir, exist_ok=True)

    dirs = ['infoVAE', 'latent-vectors', 'vis']
    for i in dirs:
        source_dir = os.path.join('results', '32-dims', i)
        shutil.copytree(source_dir, os.path.join(destination_dir, os.path.basename(source_dir)))


# Generalized entropy score: https://github.com/XixiLiu95/GEN/blob/main/benchmark.py
def gen_ood(softmax_id_val, gamma, M):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
           
        return -scores 


def df_to_gen_score(save_dir, c, gamma, M):
    prob_df = pd.read_pickle(os.path.join(save_dir, 'prob-df', c + '.pkl'))
    probs = prob_df.to_numpy()

    if c == 'xgboost':
        softmax_probs = probs
    else:
        softmax_probs = softmax(probs, axis=1)

    negative_scores = gen_ood(softmax_probs, gamma, M)

    return negative_scores




if __name__ == "__main__":
    pass 
    # copy_previous_results()