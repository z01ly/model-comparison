import h5py
import numpy as np
import os, re, shutil
from functools import reduce



def find_unreliable_idx(simulation, snapnum):
    band_list = ["morphs_g.hdf5", "morphs_i.hdf5", "morphs_r.hdf5"]
    unreliable_idx_list = []

    for band in band_list:
        if (band == "morphs_r.hdf5") and (simulation != "TNG50-1"):
            continue
        with h5py.File(os.path.join(simulation, snapnum, band), "r") as f:
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



def filter(simulation, snapnum): # discard those unreliable images
    unreliable_idx = find_unreliable_idx(simulation, snapnum)
    # print(unreliable_idx)
    subfind_ids = np.loadtxt(os.path.join(simulation, snapnum, "subfind_ids.txt"), dtype=int)
    unreliable_broadband = subfind_ids[unreliable_idx]
    print(unreliable_broadband.shape)

    destination_dir = '../' + simulation + '_' + snapnum
    os.makedirs(destination_dir, exist_ok=True)
    source_dir = '../mock_images/' + simulation + '_' + snapnum

    for filename in os.listdir(source_dir):
        match = re.match(r'broadband_(\d+)\.png', filename)
        number = int(match.group(1))
        if number not in unreliable_broadband:
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.copy2(source_path, destination_path)

    print(len(os.listdir(source_dir)) - len(os.listdir(destination_dir)))



def manual_deletion(simulation, snapnum): # TNG50-1 snapnum_099
    destination_dir = '../' + simulation + '_' + snapnum

    num_list = [51, 52, 60, 66, 79, 101]
    broken_images = ['broadband_' + str(num) + '.png' for num in num_list]

    for broken in broken_images:
        file_path = os.path.join(destination_dir, broken)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{broken}' deleted.")




if __name__ == '__main__':
    # filter("TNG50-1", "snapnum_099")
    # filter("TNG100-1", "snapnum_099")
    # filter("illustris-1", "snapnum_135")

    manual_deletion("TNG50-1", "snapnum_099")