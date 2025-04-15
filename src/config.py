from pathlib import Path

# Basic configs
LATENT_DIM = 512
K_VALUE = 32

# Paths to results
RESULTS_SAMPLING1_PATH = Path("results/sampling1")
RESULTS_SAMPLING1_UMAP_SAVE_PATH = RESULTS_SAMPLING1_PATH / "vis" / "umap" / "save"


# Paths to SDSS images
SDSS_CUTOUTS_PATH = Path("data/sdss/cutouts")
SDSS_IMAGE_PATH = Path("data/sdss-data")
SDSS_TRAIN_PATH = SDSS_IMAGE_PATH / "train" / "cutouts"
SDSS_ESVAL_PATH = SDSS_IMAGE_PATH / "esval" / "cutouts"
SDSS_VAL_PATH = SDSS_IMAGE_PATH / "val" / "cutouts"
SDSS_TEST_PATH = SDSS_IMAGE_PATH / "test" / "cutouts"


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


