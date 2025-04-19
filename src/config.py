from pathlib import Path

# ===========================================================
# General
# ===========================================================


# Basic configs
LATENT_DIM = 512
K_VALUE = 32


# Paths to results (default sampling 1)
# Change RESULTS_PATH to switch between versions
RESULTS_PATH = Path("results")
RESULTS_LATENT_VECTORS = RESULTS_PATH / "latent-vectors"
RESULTS_UMAP_SAVE_PATH = RESULTS_PATH / "vis" / "umap" / "save"
RESULTS_STREAMLIT_UMAP_PATH = RESULTS_PATH / "st-umap"


# Paths to SDSS images
SDSS_CUTOUTS_PATH = Path("data/sdss/cutouts")
SDSS_IMAGE_PATH = Path("data/sdss_data")
SDSS_TRAIN_PATH = SDSS_IMAGE_PATH / "train" / "cutouts"
SDSS_ESVAL_PATH = SDSS_IMAGE_PATH / "esval" / "cutouts"
SDSS_VAL_PATH = SDSS_IMAGE_PATH / "val" / "cutouts"
SDSS_TEST_PATH = SDSS_IMAGE_PATH / "test" / "cutouts"


# ===========================================================
# Old data paths
# ===========================================================


MOCK_TRAIN = Path("data/mock_train")
MOCK_TEST = Path("data/mock_test")


# ===========================================================
# New data preprocessing paths
# ===========================================================

# Broken images
BROKEN_TNG50_IMAGES = [1, 40, 51, 52, 60, 66, 79, 101]
BROKEN_AGNRT_IMAGES = ["AGN_g3.49e11_", "AGN_g5.53e12_"]
BROKEN_NOAGNRT_IMAGES = ["noAGN_g3.49e11_"]
BROKEN_NIHAO_IMAGES = BROKEN_AGNRT_IMAGES + BROKEN_NOAGNRT_IMAGES


# Paths to original raw data of simulation images
ILLUSTRISTNG_RAW_PATH = Path("data/raw/illustrisTNG")
TNG50_RAW_PATH = ILLUSTRISTNG_RAW_PATH / "TNG50"
TNG100_RAW_PATH = ILLUSTRISTNG_RAW_PATH / "TNG100"
NIHAORT_RAW_PATH = "data/raw/NIHAOrt"


# Paths to cleaned data of simulation images
ILLUSTRISTNG_CLEAN_PATH = Path("data/cleaned/illustrisTNG")
TNG50_BROKEN_IDX_PATH = ILLUSTRISTNG_CLEAN_PATH / "TNG50-broken-idx.pkl"
TNG100_BROKEN_IDX_PATH = ILLUSTRISTNG_CLEAN_PATH / "TNG100-broken-idx.pkl"
TNG50_CLEAN_PATH = ILLUSTRISTNG_CLEAN_PATH / "TNG50"
TNG100_CLEAN_PATH = ILLUSTRISTNG_CLEAN_PATH / "TNG100"
NIHAORT_CLEAN_PATH = "data/cleaned/NIHAOrt"


# Paths to up(down)sampled simulation images - sampling 1
# 


# Paths to up(down)sampled simulation images - sampling 2
ILLUSTRISTNG_SAMPLE_PATH_2 = Path("data/sampled/sampling2/illustrisTNG")
TNG50_SAMPLE_PATH_2 = ILLUSTRISTNG_SAMPLE_PATH_2 / "TNG50"
TNG100_SAMPLE_PATH_2 = ILLUSTRISTNG_SAMPLE_PATH_2 / "TNG100"
NIHAORT_SAMPLE_PATH_2 = "data/sampled/sampling2/NIHAOrt"


# Paths to organized simulation images - sampling 1
MOCK_ORGANIZE_PATH_1 = Path("data/organized/sampling1")


# Paths to organized simulation images - sampling 2
MOCK_ORGANIZE_PATH_2 = Path("data/organized/sampling2")


# Paths to train-test split simulation images - sampling 1
MOCK_PROCESS_PATH_1 = Path("data/processed/sampling1")
MOCK_TRAIN_PATH_1 = MOCK_PROCESS_PATH_1 / "train"
MOCK_TEST_PATH_1 = MOCK_PROCESS_PATH_1 / "test"


# Paths to train-test split simulation images - sampling 2
MOCK_PROCESS_PATH_2 = Path("data/processed/sampling2")
MOCK_TRAIN_PATH_2 = MOCK_PROCESS_PATH_2 / "train"
MOCK_TEST_PATH_2 = MOCK_PROCESS_PATH_2 / "test"

