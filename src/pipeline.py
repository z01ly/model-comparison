import os
import torch
import numpy as np

import src.pre.image_pre

import src.infoVAE.utils
import src.infoVAE.mmdVAE_train
import src.infoVAE.mmdVAE_test
import src.infoVAE.plot
import src.infoVAE.dim_meaning
import src.infoVAE.tsne_range

import src.classification.cross_val
import src.classification.test_sdss
import src.classification.cross_val_MLP
import src.classification.test_sdss_MLP
import src.classification.utils



def step1_image_pre(mock_dir_list, minority_dir_list=[]):
    """
    Suppose each directory is for a model and stored in src/data
    The directory name is the model name
    An example: 'src/data/AGNrt'
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



def step2_infoVAE(trainset_list, valset_list, model_str_list, tsne_key, compare_dict={}):
    # some parameters
    gpu_id = 1
    workers = 4
    batch_size = 500
    image_size = 64
    nc = 3
    nz = 32
    z_dim = nz
    n_filters = 64
    after_conv = src.infoVAE.utils.conv_size_comp(image_size)
    use_cuda = True

    # model
    model = src.infoVAE.mmdVAE_train.Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

    with torch.no_grad():
        # encode images to latent vectors
        test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in trainset_list]
        test_dataroots.extend([os.path.join('src/data/mock_valset', subdir) for subdir in valset_list])

        for test_dataroot in test_dataroots:
            directory_names = test_dataroot.split(os.path.sep)
            extraction = f"{directory_names[-2][5: ]}_{directory_names[-1]}"

            savefig_path = 'src/infoVAE/test_results/images_in_testing/fig_' + extraction + '.png'
            z = src.infoVAE.mmdVAE_test.test(model, test_dataroot=test_dataroot, savefig_path=savefig_path,
                    z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                    use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)

            np.save('src/infoVAE/test_results/latent/' + extraction + '.npy', z)

        # plot residual images
        folder_paths = [os.path.join('src/data/mock_trainset', subdir, 'test') for subdir in trainset_list if not subdir.endswith("times")]
        folder_paths.extend([os.path.join('src/data/mock_valset', subdir, 'test') for subdir in valset_list if not subdir.endswith("times")])
        for folder_path in folder_paths:
            src.infoVAE.plot.plot_residual(model, folder_path, gpu_id, use_cuda, False)
        src.infoVAE.plot.plot_residual(model, 'src/data/sdss_data/test/cutouts/', gpu_id, use_cuda, True)

        for model_str in model_str_list:
            # check the meaning of each dimension of latent vectors
            src.infoVAE.dim_meaning.main(model_str, model, gpu_id, use_cuda=True, dimension=32)

            # check range of latent vectors of each model
            src.infoVAE.dim_meaning.check_range(model_str)

    # compute tsne results and save
    # src.infoVAE.tsne_range.save_tsne_results(model_list=[], is_sdss_test=True)
    src.infoVAE.tsne_range.save_tsne_results(model_str_list)

    # plot tsne results and compare with sdss tsne result
    src.infoVAE.tsne_range.plot_tsne(tsne_key, model_str_list)
    src.infoVAE.tsne_range.plot_tsne(tsne_key, model_str_list, include_sdss=True)

    # optional: plot compare results between two sets of models
    if compare_dict != {}:
        src.infoVAE.tsne_range.plot_compare(compare_dict)



def step3_classification(key, model_names):
    # make directories
    src.classification.utils.pre_makedirs(key)

    # load useful data
    X, y = src.classification.utils.load_data_train(model_names)
    sdss_test_data = np.load('src/infoVAE/test_results/latent/sdss_test.npy')
    print(sdss_test_data.shape)

    # cross validation
    src.classification.cross_val.main(key, [s.split('_')[0] for s in model_names], X, y, 'integer', 'random-forest')
    src.classification.cross_val.main(key, [s.split('_')[0] for s in model_names], X, y, 'integer', 'xgboost')

    classifier_keys = ['single-MLP', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    for classifier_key in classifier_keys:
        src.classification.cross_val_MLP.main(key, [s.split('_')[0] for s in model_names], X, y, classifier_key, max_iter=500)

    # test on sdss
    src.classification.test_sdss.train(key, 'random-forest', X, y)
    src.classification.test_sdss.train(key, 'xgboost', X, y)
    src.classification.test_sdss.test(key, [s.split('_')[0] for s in model_names], 'random-forest', sdss_test_data)
    src.classification.test_sdss.test(key, [s.split('_')[0] for s in model_names], 'xgboost', sdss_test_data)

    for classifier_key in classifier_keys:
        scaler = src.classification.test_sdss_MLP.train(key, classifier_key, X, y, max_iter=500)
        src.classification.test_sdss_MLP.test(scaler, key, [s.split('_')[0] for s in model_names], classifier_key, sdss_test_data)




if __name__ == '__main__':
    # keep a copy of original images in directory mock_images before preprocessing

    """
    mock_dir_list = ['src/data/AGNrt', 'src/data/NOAGNrt', 'src/data/UHDrt', 'src/data/n80rt']
    # the number of images in each model
    for mock_dir in mock_dir_list:
        print(f"{mock_dir}: {len(os.listdir(mock_dir))}")
    minority_dir_list = ['src/data/AGNrt', 'src/data/NOAGNrt', 'src/data/UHDrt', 'src/data/n80rt']
    step1_image_pre(mock_dir_list, minority_dir_list)
    # manually move TNG data since they have been processed
    """

    """
    trainset_list = ['AGNrt', 'NOAGNrt', 'UHDrt', 'n80rt', 'AGNrt_2times', 'NOAGNrt_2times', 'UHDrt_2times', 'n80rt_2times']
    valset_list = ['AGNrt', 'NOAGNrt', 'UHDrt', 'n80rt']
    # model_str_list = ['AGNrt', 'NOAGNrt', 'UHDrt', 'n80rt', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099', 'illustris-1_snapnum_135']
    # model_str_list = ['AGNrt', 'NOAGNrt', 'UHDrt', 'n80rt']
    # tsne_key = 'NIHAOrt'
    model_str_list = ['TNG100-1_snapnum_099', 'illustris-1_snapnum_135', 'TNG50-1_snapnum_099']
    tsne_key = 'illustris'
    compare_dict = {'NIHAOrt': ['AGNrt', 'NOAGNrt', 'UHDrt', 'n80rt'], 'TNG': ['TNG100-1_snapnum_099', 'TNG50-1_snapnum_099']}
    step2_infoVAE(trainset_list, valset_list, model_str_list, tsne_key, compare_dict=compare_dict)
    """

    key = 'NIHAOrt_TNG'
    model_names = ['AGNrt_2times', 'NOAGNrt_2times', 'TNG100-1_snapnum_099', 'TNG50-1_snapnum_099_2times', 'UHDrt_2times', 'n80rt_2times']
    step3_classification(key, model_names)
