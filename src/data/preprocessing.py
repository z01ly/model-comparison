import os
import pickle
import random
import shutil
import h5py
import re
import numpy as np
import scipy
import cv2

from PIL import Image
from functools import reduce

import src.config as config



# ===========================================================
# Part 1: Delete Broken Simulation Images
# ===========================================================


class FilterMorphFlag():
    """
    Clean IllustrisTNG images
    """
    def __init__(self, simulation, snapnum):
        self.simulation = simulation
        self.snapnum = snapnum

    
    def find_unreliable_idx(self, hdf5_path):
        """
        Find indices of unreliable data according to IllustrisTNG website info
        """
        band_list = ["morphs_g.hdf5", "morphs_i.hdf5", "morphs_r.hdf5"]
        unreliable_idx_list = []

        for band in band_list:
            if (band == "morphs_r.hdf5") and (self.simulation != "TNG50"):
                continue
            with h5py.File(os.path.join(hdf5_path, self.simulation, self.snapnum, band), "r") as f:
                flag_dataset = f['flag']
                flag_data = flag_dataset[()]
                # print(f"flag_data shape of {band}: {flag_data.shape}")

                sn_dataset = f['sn_per_pixel']
                sn_data = sn_dataset[()]
                # print(f"sn_data shape of {band}: {sn_data.shape}")

            unreliable_flag_idx = np.where(flag_data == 1)[0]
            # print(f"unreliable_flag_idx of {band} shape: {unreliable_flag_idx.shape}")
            unreliable_idx_list.append(unreliable_flag_idx)

            unreliable_sn_idx = np.where(sn_data <= 2.5)[0]
            # print(f"unreliable_sn_idx of {band} shape: {unreliable_sn_idx.shape}")
            unreliable_idx_list.append(unreliable_sn_idx)

        # print(f"len of unreliable_idx_list: {len(unreliable_idx_list)}")
        union_result = reduce(np.union1d, unreliable_idx_list)
        # print(f"union_result shape: {union_result.shape}")

        return union_result


    def filter(self, source_dir, destination_dir, hdf5_path=config.ILLUSTRISTNG_RAW_PATH): 
        """
        Discard unreliable images
        And copy reliable images from source directory to destination directory
        """
        unreliable_idx = self.find_unreliable_idx(hdf5_path)
        # print(unreliable_idx)
        subfind_ids = np.loadtxt(os.path.join(hdf5_path, self.simulation, self.snapnum, "subfind_ids.txt"), dtype=int)
        unreliable_broadband = subfind_ids[unreliable_idx]
        print(unreliable_broadband.shape)

        # source_dir = os.path.join('../mock-images', 'illustris', self.simulation)
        # destination_dir = os.path.join('data', self.simulation)
        os.makedirs(destination_dir, exist_ok=True)

        for filename in os.listdir(source_dir):
            match = re.match(r'broadband_(\d+)\.png', filename)
            number = int(match.group(1))
            if number not in unreliable_broadband:
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, filename)
                shutil.copy2(source_path, destination_path)

        print(len(os.listdir(source_dir)) - len(os.listdir(destination_dir)))



def deletion_TNG50(destination_dir):
    """
    Only used for TNG50
    A few broken images are not filterd out by checking flag 
    """
    num_list = config.BROKEN_TNG50_IMAGES
    broken_images = ["broadband_" + str(num) + ".png" for num in num_list]

    for broken in broken_images:
        file_path = os.path.join(destination_dir, broken)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{broken}' deleted.")
        else:
            print(f"Not found: {file_path}")



def deletion_IllustrisTNG(broken_idx_path: str, TNG_path: str) -> None:
    """
    Delete broken IllustrisTNG images (using stored indices, without hdf5 files)
    """
    with open(broken_idx_path, 'rb') as f:
        broken_indices = pickle.load(f)
    print(f"Length of indices list: {len(broken_indices)}")
    
    for idx in broken_indices:
        image_path = os.path.join(TNG_path, f"broadband_{idx}.png")  
        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            print(f"Not found: {image_path}")



def deletion_NIHAOrt() -> None:
    """
    Delete broken NIHAOrt images
    """
    galaxy_patterns = config.BROKEN_NIHAO_IMAGES

    for pattern in galaxy_patterns:
        for i in range(20):
            filename = f"{pattern}{i:02d}.png" 
            file_path = os.path.join(config.NIHAORT_CLEAN_PATH, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"Not found: {file_path}")




# ===========================================================
# Part 2: Down/Upsampling of Simulation Images
# ===========================================================


def cubic_sampling(folder_path: str) -> None:
    """
    Cubic interpolation for downsampling and upsampling
    """
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        image_array = np.array(image)

        # sample the image to 64x64 using scipy.ndimage.zoom
        if image_array.shape[0] != 64:
            downsampled_array = scipy.ndimage.zoom(image_array, (64 / image_array.shape[0], 64 / image_array.shape[1], 1), order=3)
            downsampled_image = Image.fromarray(downsampled_array.astype(np.uint8))
        else:
            continue

        # save the sampled image, overwriting the original file
        downsampled_image.save(image_path)



def area_cubic_sampling(folder_path: str) -> None:
    """
    INTER_AREA for downsampling (if image is larger than 64x64)
    INTER_CUBIC for upsampling (if image is smaller than 64x64)
    Important: BGR to RGB conversion and save images in RGB mode
    """
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path) # discard the alpha (A) channel, BGR mode
        h, _ = img.shape[:2]

        if h < 64:
            interpolation = cv2.INTER_CUBIC
        elif h > 64:
            interpolation = cv2.INTER_AREA
        else:
            continue

        resized_img = cv2.resize(img, (64, 64), interpolation=interpolation) # BGR mode
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB) # RGB mode 
        img_pil = Image.fromarray(resized_img_rgb)
        img_pil.save(img_path)
        # cv2.imwrite(img_path, resized_img_rgb)
    print(f"Finished: {folder_path}")




def organize_NIHAO_images(source_dir: str, target_root: str) -> None:
    """
    Copy NIHAO images: create a folder for each class
    TNG50 and TNG100 already have their own folders, just copy them.
    """
    class_map={"AGN": "AGNrt", "noAGN": "NOAGNrt", "UHD": "UHDrt", "n80": "n80rt"}

    for _, folder_name in class_map.items():
        target_path = os.path.join(target_root, folder_name)
        os.makedirs(target_path, exist_ok=True)
    
    for filename in os.listdir(source_dir):
        for class_prefix, folder_name in class_map.items():
            if filename.startswith(class_prefix):
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(target_root, folder_name, filename)
                shutil.copy2(source_path, destination_path)
                # print(f"Moved {filename} to {folder_name}")
                break



# ===========================================================
# Part 3: Train-Test Split of Simulation Images
# ===========================================================


def mock_split(source_root: str,
               train_root: str,
               test_root: str,
               model_str: str,
               rate: float=0.85) -> None:
    """
    Split simulation images to training set and test set.
    """
    train_dir = os.path.join(train_root, model_str, 'class_0')
    test_dir = os.path.join(test_root, model_str, 'class_0')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    source_dir = os.path.join(source_root, model_str)
    files_in_source_directory = os.listdir(source_dir)

    total_files = len(files_in_source_directory)
    train_count = int(total_files * rate)
    # test_count = total_files - train_count

    random.shuffle(files_in_source_directory)

    train_files = files_in_source_directory[:train_count]
    test_files = files_in_source_directory[train_count:]

    for file in train_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(train_dir, file)
        shutil.copy2(source_path, dest_path)

    for file in test_files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(test_dir, file)
        shutil.copy2(source_path, dest_path)

    print(f"current model: {model_str}")
    print(f"{len(os.listdir(train_dir))} files copied to the training set.")
    print(f"{len(os.listdir(test_dir))} files copied to the test set.\n")




# ===========================================================
# Part 4: SDSS Processing
# ===========================================================


def sdss_split(source_folder=config.SDSS_CUTOUTS_PATH):
    """
    Split SDSS images to train, validation and test set
    """
    os.makedirs(config.SDSS_TRAIN_PATH, exist_ok=True)
    os.makedirs(config.SDSS_ESVAL_PATH, exist_ok=True)
    os.makedirs(config.SDSS_VAL_PATH, exist_ok=True)
    os.makedirs(config.SDSS_TEST_PATH, exist_ok=True)

    image_files = os.listdir(source_folder)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_ratio = 0.7
    esval_ratio = 0.1
    val_ratio = 0.1
    # test_ratio = 0.1
    num_train = int(total_images * train_ratio)
    num_esval = int(total_images * esval_ratio)
    num_val = int(total_images * val_ratio)
    # num_test = total_images - num_train - num_esval - num_val

    train_files = image_files[: num_train]
    esval_files = image_files[num_train: num_train + num_esval]
    val_files = image_files[num_train + num_esval: num_train + num_esval + num_val]
    test_files = image_files[num_train + num_esval + num_val: ]

    for file_ in train_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_TRAIN_PATH, file_))

    for file_ in esval_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_ESVAL_PATH, file_))

    for file_ in val_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_VAL_PATH, file_))

    for file_ in test_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(config.SDSS_TEST_PATH, file_))




if __name__ == '__main__':
    # sdss_split()
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    # mock_data_pre(model_str_list, 64)
    # mock_data_count(model_str_list)
    
    # deletion_IllustrisTNG(config.TNG50_BROKEN_IDX_PATH, config.TNG50_CLEAN_PATH)
    # deletion_IllustrisTNG(config.TNG100_BROKEN_IDX_PATH, config.TNG100_CLEAN_PATH)
    # deletion_NIHAOrt()
    
    # area_cubic_sampling(config.TNG50_SAMPLE_PATH_2)
    # area_cubic_sampling(config.TNG100_SAMPLE_PATH_2)
    # area_cubic_sampling(config.NIHAORT_SAMPLE_PATH_2)
    # organize_NIHAO_images(config.NIHAORT_SAMPLE_PATH_2, config.MOCK_ORGANIZE_PATH_2)
    
    # for model_str in model_str_list: 
    #     mock_split(config.MOCK_ORGANIZE_PATH_1, config.MOCK_TRAIN_PATH_1, config.MOCK_TEST_PATH_1, model_str)

    # for model_str in model_str_list: 
    #     mock_split(config.MOCK_ORGANIZE_PATH_2, config.MOCK_TRAIN_PATH_2, config.MOCK_TEST_PATH_2, model_str)

