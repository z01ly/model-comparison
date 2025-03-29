# config.py
from pathlib import Path

# Results path
RESULTS_PATH = Path("results")


# Path to SDSS images
SDSS_CUTOUTS_PATH = Path("data/sdss/cutouts")
SDSS_IMAGE_PATH = Path("data/sdss_data")
SDSS_TRAIN_PATH = SDSS_IMAGE_PATH / "train" / "cutouts"
SDSS_ESVAL_PATH = SDSS_IMAGE_PATH / "esval" / "cutouts"
SDSS_VAL_PATH = SDSS_IMAGE_PATH / "val" / "cutouts"
SDSS_TEST_PATH = SDSS_IMAGE_PATH / "test" / "cutouts"


# Paths to original raw data of simulated images
ILLUSTRISTNG_RAW_PATH = Path("data/raw/illustrisTNG")
ILLUSTRIS_RAW_PATH = ILLUSTRISTNG_RAW_PATH / "illustris"
TNG50_RAW_PATH = ILLUSTRISTNG_RAW_PATH / "TNG50"
TNG100_RAW_PATH = ILLUSTRISTNG_RAW_PATH / "TNG100"

NIHAO_RAW_PATH = "data/raw/NIHAOrt"

# Paths to split simulation images

