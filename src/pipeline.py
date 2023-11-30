import os

import src.pre.image_pre



def step0_image_pre(mock_dir_list, minority_dir_list=[]):
    """
    Suppose each directory is for a model and stored in src/data
    An example: ['src/data/AGNrt', 'src/data/NOAGNrt', 'src/data/UHDrt', 'src/data/n80rt']
    """

    # check size of sdss data
    print("\n")
    sdss_data_path = 'src/data/sdss_data/test/cutouts'
    sdss_data_size = src.pre.image_pre.check_image_size(sdss_data_path)

    
    for mock_dir_path in mock_dir_list:
        print("\n")
        # check size of mock images
        mock_data_size = src.pre.image_pre.check_image_size(mock_dir_path)

        # upsample or downsample mock images if the size doesn't match sdss size
        if mock_data_size != sdss_data_size:
            src.pre.image_pre.sample_mock(mock_dir_path)

        # split mock images to trainset and valset (valset is used later in ood)
        src.pre.image_pre.mock_split(mock_dir_path)

        # add a subdir named 'test' to prepare the directory for infoVAE dataloader
        src.pre.image_pre.add_subdir_move_files('src/data/mock_trainset/' + mock_dir_path[9: ], 'test')
        src.pre.image_pre.add_subdir_move_files('src/data/mock_valset/' + mock_dir_path[9: ], 'test')

    # oversample minority models once
    if minority_dir_list != []:
        for minority_path in minority_dir_list:
            print("\n")
            source = 'src/data/mock_trainset/' + minority_path[9: ] + '/test'
            destination = 'src/data/mock_trainset/' + minority_path[9: ] +'_2times/test'
            src.pre.image_pre.oversample_minority(source, destination, 2)








if __name__ == '__main__':
    # keep a copy of original images somewhere else before preprocessing
    mock_dir_list = ['src/data/AGNrt', 'src/data/NOAGNrt', 'src/data/UHDrt', 'src/data/n80rt']
    # the number of images in each model
    for mock_dir in mock_dir_list:
        print(f"{mock_dir}: {len(os.listdir(mock_dir))}")
    minority_dir_list = ['src/data/AGNrt', 'src/data/NOAGNrt', 'src/data/UHDrt', 'src/data/n80rt']
    step0_image_pre(mock_dir_list, minority_dir_list)
    # manually move TNG data since they have been processed