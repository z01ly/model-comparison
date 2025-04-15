import os
import pickle
import pandas as pd

from src.data.utils import copy_df_path_images
import src.config as config


# ===========================================================
# UMAP related functions
# ===========================================================


def print_umap_example(umap_save_path: str, model_or_sdss: str) -> None:
    """
    Print the structure of UMAP saved pickle files
    """
    current_pkl = os.path.join(umap_save_path, model_or_sdss + '.pkl')
    with open(current_pkl, 'rb') as f:
        pkl_data = pickle.load(f)

    print(f"Number of columns: {pkl_data.shape[1]}")
    print(f"Head and examples:\n {pkl_data.head(5)}")


# ===========================================================
# For both tSNE and UMAP
# ===========================================================


def reduce_sdss(savepath_prefix: str, sdss_frac: float=0.5) -> None:
    os.makedirs(os.path.join('results', 'streamlit', 'sdss-test-reduced'), exist_ok=True)

    df = pd.read_pickle(os.path.join(savepath_prefix, 'vis', 'tsne', 'embedded-z.pkl'))
    print(df.shape)

    sdss_data = df[df['label'] == 'sdss']
    non_sdss_data = df[df['label'] != 'sdss']

    sdss_sampled = sdss_data.sample(frac=sdss_frac, random_state=0)
    sdss_sampled.to_pickle(os.path.join('results', 'streamlit', 'sdss-sampled.pkl'))

    # copy images
    copy_df_path_images(os.path.join('results', 'streamlit'), os.path.join('results', 'streamlit', 'sdss-test-reduced'), 'sdss-sampled')

    df_reduced = pd.concat([sdss_sampled, non_sdss_data], axis=0)
    df_reduced.reset_index(drop=True, inplace=True)
    print(df_reduced.shape)

    df_reduced.to_pickle(os.path.join('results', 'streamlit', 'embedded-z-reduced.pkl'))




if __name__ == '__main__':
    # nz = 32
    # savepath_prefix = 'results/' + str(nz) + '-dims'
    #  model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # reduce_sdss(savepath_prefix)

    # print_umap_example(config.RESULTS_SAMPLING1_UMAP_SAVE_PATH, "AGNrt")

    pkl_data = pd.read_pickle(os.path.join(config.RESULTS_SAMPLING1_PATH, "vis", "umap", "n80rt.pkl"))
    # df.to_pickle(os.path.join(config.RESULTS_SAMPLING1_PATH, "vis", "umap", "AGNrt.pkl"))
    print(f"Number of columns: {pkl_data.shape[1]}")
    print(f"Head and examples:\n {pkl_data.head(5)}")