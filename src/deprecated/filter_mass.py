import numpy as np
import os


def filter(simulation, snapnum):
    filter_mass_ids = np.load('src/illustris_make_filter/api_filter_mass_id/' + simulation + '_' + snapnum + '.npy')
    subfind_ids = np.loadtxt(os.path.join('src', 'illustris_make_filter', simulation, snapnum, 'subfind_ids.txt'), dtype=int)
    print(type(filter_mass_ids[0]))
    print(type(subfind_ids[0]))

    common_elements = np.intersect1d(filter_mass_ids, subfind_ids)
    print(len(subfind_ids))
    print(len(common_elements))

    delete_elements = np.setdiff1d(subfind_ids, common_elements)
    print(len(delete_elements))
    print(delete_elements)





if __name__ == '__main__':
    pass
    # filter("TNG50-1", "snapnum_099")
    # filter("TNG100-1", "snapnum_099")
    # filter("illustris-1", "snapnum_135")