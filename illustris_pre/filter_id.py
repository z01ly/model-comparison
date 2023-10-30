import numpy as np


def filter(simulation):
    filter_mass_ids = np.load('api_filter_mass_id/' + simulation + '.npy')
    subfind_ids = np.loadtxt('subfind_ids/' + simulation + '.txt', dtype=int)
    print(type(filter_mass_ids[0]))
    print(type(subfind_ids[0]))

    common_elements = np.intersect1d(filter_mass_ids, subfind_ids)
    print(len(subfind_ids))
    print(len(common_elements))

    delete_elements = np.setdiff1d(subfind_ids, common_elements)
    print(len(delete_elements))
    print(delete_elements)

def main():
    filter('TNG50_1_snap99')


if __name__ == '__main__':
    main()