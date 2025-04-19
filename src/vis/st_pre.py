import os
import pandas as pd

from src.data.utils import copy_df_path_images
import src.config as config


# ===========================================================
# UMAP related functions
# ===========================================================


def umap_pkl_example(filename: str) -> None:
    """
    Print the structure of UMAP saved pandas dataframe
    """
    pkl_data = pd.read_pickle(os.path.join(config.RESULTS_UMAP_SAVE_PATH, filename + '.pkl'))

    print(f"Number of rows: {pkl_data.shape[0]}")
    print(f"Number of columns: {pkl_data.shape[1]}")
    print(f"Head and examples:\n {pkl_data.head(5)}")



def umap_pkl_add_filename_sdss() -> None:
    """
    Mapping umap embedded points and corresponding sdss images, ref class UmapVis()
    Previous pandas df head: f0, f1, label
    Now df head: f0, f1, filename, label
    """
    sdss_df = pd.read_pickle(os.path.join(config.RESULTS_LATENT_VECTORS, 'sdss', 'test.pkl'))
    sdss_umap_df_path = os.path.join(config.RESULTS_UMAP_SAVE_PATH, 'sdss.pkl')
    sdss_umap_df = pd.read_pickle(sdss_umap_df_path)

    sdss_umap_df['filename'] = sdss_df['filename'].values
    sdss_umap_df.to_pickle(sdss_umap_df_path)



def umap_pkl_add_filename_mock(model_str_list: list[str]) -> None:
    """
    Mapping umap embedded points and corresponding simulation images, ref class UmapVis()
    Previous pandas df head: f0, f1, label
    Now df head: f0, f1, filename, label
    """
    for model_str in model_str_list:
        mock_train_df = pd.read_pickle(os.path.join(config.RESULTS_LATENT_VECTORS, 'train', model_str + '.pkl'))
        mock_test_df = pd.read_pickle(os.path.join(config.RESULTS_LATENT_VECTORS, 'test', model_str + '.pkl'))
        mock_df = pd.concat([mock_train_df, mock_test_df], axis=0)
        mock_df.reset_index(drop=True, inplace=True)

        mock_umap_df_path = os.path.join(config.RESULTS_UMAP_SAVE_PATH, model_str + '.pkl')
        mock_umap_df = pd.read_pickle(mock_umap_df_path)

        mock_umap_df['filename'] = mock_df['filename'].values
        mock_umap_df.to_pickle(mock_umap_df_path)



def umap_delete_broken_TNG50() -> None:
    """
    Delete df rows corresponding to broken images
    """
    df_path = os.path.join(config.RESULTS_UMAP_SAVE_PATH, "TNG50.pkl")
    df = pd.read_pickle(df_path)
    print("Number of rows:", df.shape[0])

    filenames_to_remove = [f"broadband_{n}.png" for n in config.BROKEN_TNG50_IMAGES]
    df["basename"] = df["filename"].str.split("/").str[-1]
    existing_filenames = df[df["basename"].isin(filenames_to_remove)]
    print("These rows exist in the DataFrame and will be removed:")
    print(existing_filenames["filename"])

    df_cleaned = df[~df["basename"].isin(filenames_to_remove)] # drop the rows
    df_cleaned = df_cleaned.drop(columns=["basename"])
    df_cleaned.to_pickle(df_path)
    print("Number of rows after deletion:", pd.read_pickle(df.path).shape[0])



def umap_delete_broken_NIHAO(model_str: str, prefixes_to_remove: list[str]) -> None:
    """
    Delete df rows corresponding to broken images
    """
    df_path = os.path.join(config.RESULTS_UMAP_SAVE_PATH, model_str + '.pkl')
    df = pd.read_pickle(df_path)
    print("Number of rows:", df.shape[0])

    df["basename"] = df["filename"].str.split("/").str[-1]
    matches = df[df['basename'].str.startswith(tuple(prefixes_to_remove))]
    print("These rows will be removed:")
    print(matches)

    df_cleaned = df[~df['basename'].str.startswith(tuple(prefixes_to_remove))]
    df_cleaned = df_cleaned.drop(columns=["basename"])
    df_cleaned.to_pickle(df_path)
    print("Number of rows after deletion:", pd.read_pickle(df_path).shape[0])



def umap_stack(model_str_list: list[str]) -> None:
    """
    Stack umap pkl files for creating the st app
    """
    dfs = []
    for f in model_str_list:
        file_path = os.path.join(config.RESULTS_UMAP_SAVE_PATH, f + '.pkl')
        df = pd.read_pickle(file_path)
        dfs.append(df)

    sdss_sample_path = os.path.join(config.RESULTS_STREAMLIT_UMAP_PATH, 'sdss-sampled.pkl')
    sdss_sample_df = pd.read_pickle(sdss_sample_path)
    dfs.append(sdss_sample_df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_pickle(os.path.join(config.RESULTS_STREAMLIT_UMAP_PATH, 'embedded.pkl'))



def umap_sample_sdss(model_str_list: list[str], frac: float) -> None:
    """
    Sample sdss test set images for creating the st app
    """
    os.makedirs(os.path.join(config.RESULTS_STREAMLIT_UMAP_PATH, 'sdss-test-sampled')) # for saving images

    sdss_df = pd.read_pickle(os.path.join(config.RESULTS_UMAP_SAVE_PATH, 'sdss.pkl'))
    sdss_sampled = sdss_df.sample(frac=frac, random_state=0)
    sdss_sampled.to_pickle(os.path.join(config.RESULTS_STREAMLIT_UMAP_PATH, 'sdss-sampled.pkl'))

    umap_stack(model_str_list)

    copy_df_path_images(config.RESULTS_STREAMLIT_UMAP_PATH, \
                        os.path.join(config.RESULTS_STREAMLIT_UMAP_PATH, 'sdss-test-sampled'), \
                        'sdss-sampled')



# ===========================================================
# tSNE
# ===========================================================


def tsne_sample_sdss(savepath_prefix: str, sdss_frac: float=0.5) -> None:
    os.makedirs(os.path.join('results', 'streamlit', 'sdss-test-reduced'), exist_ok=True)

    df = pd.read_pickle(os.path.join(savepath_prefix, 'vis', 'tsne', 'embedded-z.pkl'))
    print(df.shape)

    sdss_data = df[df['label'] == 'sdss']
    non_sdss_data = df[df['label'] != 'sdss']

    sdss_sampled = sdss_data.sample(frac=sdss_frac, random_state=0)
    sdss_sampled.to_pickle(os.path.join('results', 'streamlit', 'sdss-sampled.pkl'))

    # copy images
    copy_df_path_images(os.path.join('results', 'streamlit'), \
                        os.path.join('results', 'streamlit', 'sdss-test-reduced'), \
                        'sdss-sampled')

    df_reduced = pd.concat([sdss_sampled, non_sdss_data], axis=0)
    df_reduced.reset_index(drop=True, inplace=True)
    print(df_reduced.shape)

    df_reduced.to_pickle(os.path.join('results', 'streamlit', 'embedded-z-reduced.pkl'))




if __name__ == '__main__':
    # nz = 32
    # savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # reduce_sdss(savepath_prefix)

    # umap_pkl_add_filename_sdss()
    # umap_pkl_add_filename_mock(model_str_list)
    # umap_pkl_example("AGNrt")

    # umap_delete_broken_TNG50()
    # umap_delete_broken_NIHAO('AGNrt', config.BROKEN_AGNRT_IMAGES)
    # umap_delete_broken_NIHAO('NOAGNrt', config.BROKEN_NOAGNRT_IMAGES)

    # umap_sample_sdss(model_str_list, 0.3)

    