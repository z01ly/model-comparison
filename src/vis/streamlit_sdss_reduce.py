import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')

from src.data.utils import copy_df_path_images


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
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    # model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    reduce_sdss(savepath_prefix)
