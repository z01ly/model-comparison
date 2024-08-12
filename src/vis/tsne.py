import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def tsne_save(savepath_prefix, nz, model_str_dict):
    dfs = []

    sdss_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl'))
    sdss_df['label'] = 'sdss'
    # print(sdss_df.head())
    sampled_sdss_df = sdss_df.sample(frac=0.6, random_state=42)
    dfs.append(sampled_sdss_df)

    for model_str, frac_value in model_str_dict.items():
        for key in ['train', 'test']:
            z_df = pd.read_pickle(os.path.join(savepath_prefix, 'latent-vectors', key, model_str + '.pkl'))
            z_df['label'] = model_str
            sampled_z_df = z_df.sample(frac=frac_value, random_state=42)
            dfs.append(sampled_z_df)
    
    latent_z_df = pd.concat(dfs, axis=0)
    latent_z_df.reset_index(drop=True, inplace=True)
    print(latent_z_df.shape)
    filename_col = latent_z_df.iloc[:, -2].to_numpy()
    label_col = latent_z_df.iloc[:, -1].to_numpy()

    latent_z = latent_z_df.iloc[:, 0:nz].to_numpy()
    standardized_z = StandardScaler().fit_transform(latent_z)
    embedded_z = TSNE(n_components=2, perplexity=100, init='pca', random_state=42).fit_transform(standardized_z)
    embedded_z_df = pd.DataFrame(embedded_z, columns=[f'f{i}' for i in range(2)])
    embedded_z_df['filename'] = filename_col
    embedded_z_df['label'] = label_col

    embedded_z_df.to_pickle(os.path.join(savepath_prefix, 'vis', 'tsne', 'embedded-z.pkl'))



def plot_tsne(savepath_prefix, model_str_list):
    embedded_z = pd.read_pickle(os.path.join(savepath_prefix, 'vis', 'tsne', 'embedded-z.pkl'))
    sns.scatterplot(data=embedded_z, x='f0', y='f1', hue='label', palette="bright", s=5)
    plt.savefig(os.path.join(savepath_prefix, 'vis', 'tsne', 'plot-mix.png'))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    for ax, model_str in zip(axs.flat, model_str_list):
        subset_df = embedded_z[embedded_z['label'].isin(['sdss', model_str])]
        sns.scatterplot(x='f0', y='f1', data=subset_df, ax=ax, hue='label', s=5)
        ax.set_title(f'sdss and {model_str}')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath_prefix, 'vis', 'tsne', 'plot-separate.png'))

