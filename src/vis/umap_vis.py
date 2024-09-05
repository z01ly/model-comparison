import umap

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


class UmapVis():
    def __init__(self, savepath_prefix, nz, model_str_list, sdss_df_path, key='umap'):
        self.savepath_prefix = savepath_prefix
        self.nz = nz
        self.model_str_list = model_str_list
        self.sdss_df_path = sdss_df_path

        self.path_save_embedding = os.path.join(savepath_prefix, 'vis', key, 'save')
        self.path_plot = os.path.join(savepath_prefix, 'vis', key, 'plot')
    

    def embedding_save(self):
        os.makedirs(self.path_save_embedding, exist_ok=True)

        sdss_df = pd.read_pickle(self.sdss_df_path) # .sample(frac=self.sdss_frac, random_state=0)
        sdss = sdss_df.iloc[:, 0:self.nz].to_numpy()

        scaler = StandardScaler()
        sdss_scaled = scaler.fit_transform(sdss)

        # NotImplementedError: Transforming data into an existing embedding not supported for densMAP.
        reducer = umap.UMAP(n_neighbors=50, n_components=2, random_state=0, verbose=False)

        print("Computing sdss embedding...")
        sdss_embedding = reducer.fit_transform(sdss_scaled)
        sdss_embedding_df = pd.DataFrame(sdss_embedding, columns=['f0', 'f1'])
        sdss_embedding_df['label'] = 'sdss'
        sdss_embedding_df.to_pickle(os.path.join(self.path_save_embedding, 'sdss.pkl'))

        for model_str in self.model_str_list:
            mock_train_df = pd.read_pickle(os.path.join(self.savepath_prefix, 'latent-vectors', 'train', model_str + '.pkl'))
            mock_test_df = pd.read_pickle(os.path.join(self.savepath_prefix, 'latent-vectors', 'test', model_str + '.pkl'))
            mock_df = pd.concat([mock_train_df, mock_test_df], axis=0)
            mock_df.reset_index(drop=True, inplace=True)
            mock = mock_df.iloc[:, 0:self.nz].to_numpy()

            mock_scaled = scaler.transform(mock)
            print(f"Computing {model_str} embedding...")
            mock_embedding = reducer.transform(mock_scaled)
            mock_embedding_df = pd.DataFrame(mock_embedding, columns=['f0', 'f1'])
            mock_embedding_df['label'] = model_str
            mock_embedding_df.to_pickle(os.path.join(self.path_save_embedding, model_str + '.pkl'))


    def embedding_plot(self):
        os.makedirs(self.path_plot, exist_ok=True)

        sdss_embedding_df = pd.read_pickle(os.path.join(self.path_save_embedding, 'sdss.pkl'))

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        for ax, model_str in zip(axs.flat, self.model_str_list):
            mock_embedding_df = pd.read_pickle(os.path.join(self.path_save_embedding, model_str + '.pkl'))
            subset_df = pd.concat([sdss_embedding_df, mock_embedding_df], axis=0)
            subset_df.reset_index(drop=True, inplace=True)

            sns.scatterplot(x='f0', y='f1', data=subset_df, ax=ax, hue='label', s=5)
            ax.set_title(f'sdss and {model_str}', fontsize=22)

            # Adjust font size and marker size
            ax.set_xlabel('f0', fontsize=16)
            ax.set_ylabel('f1', fontsize=16)
            
            legend = ax.legend(handletextpad=1, markerscale=3)  # markerscale controls marker size in legend
            legend.set_title('label', prop={'size': 16}) # legend title size
            plt.setp(legend.get_texts(), fontsize=16)  # legend text size

            ax.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(self.path_plot, 'plot-separate.png'))

