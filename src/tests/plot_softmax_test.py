import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.special import softmax

import src.config as config


def compute_softmax(prob_df_dir: str, c: str) -> np.ndarray:
    prob_df = pd.read_pickle(os.path.join(prob_df_dir, 'prob-df', c + '.pkl'))
    probs = prob_df.to_numpy()
    softmax_probs = softmax(probs, axis=1)

    # return softmax_probs
    return probs



def plot_softmax(c: str,
                 legend_loc: str="upper right",
                 savepath: str=config.RESULTS_PLOT_SOFTMAX_TEST,
                 sdss_dir: str=config.RESULTS_CLASSIFICATION,
                 id_dir: str=config.RESULTS_CLASSIFY_ID) -> None:
    os.makedirs(savepath, exist_ok=True)

    sdss_softmax_probs = compute_softmax(sdss_dir, c)
    ID_softmax_probs = compute_softmax(id_dir, c)
    
    colors = sns.color_palette("tab10", len(config.MODEL_STR_LIST))
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(config.MODEL_STR_LIST):
        sns.kdeplot(sdss_softmax_probs[:, i], label=f'SDSS - {label}', linestyle='-', color=colors[i])
        # sns.kdeplot(ID_softmax_probs[:, i], label=f'ID - {label}', linestyle='--', color=colors[i])
    plt.title('Distribution of Softmax Probabilities for Each Class', fontsize=20)
    plt.xlabel('Softmax Probability', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.legend(handletextpad=1, markerscale=2, fontsize=16, loc=legend_loc)
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, c + '-probs.png'))
    plt.close()



if __name__ == "__main__":
    for c in config.CLASSIFIER_LIST:
        plot_softmax(c)
