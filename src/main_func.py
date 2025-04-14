import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
import yaml

import src.main.oversampling as oversampling
import src.main.classifier as classifier 
from src.ood import GenOod
import src.main.img_encoder as img_encoder
import src.main.latent_vis as latent_vis
import src.main.xai_func as xai_func


def oversample_sim(savepath_prefix, model_str_list, minority_str_list):
    df_dir = os.path.join(savepath_prefix, 'latent-vectors', 'train')
    base_dir = os.path.join(savepath_prefix, 'oversampling')
    image_dir = os.path.join(base_dir, 'images')
    oversampled_image_dir = os.path.join(base_dir, 'oversampled-images')
    oversampled_vector_dir = os.path.join(base_dir, 'oversampled-vectors')
    oversampled_dense_vector_dir = os.path.join(base_dir, 'oversampled-dense-vectors')

    oversampling.img_copy(savepath_prefix, model_str_list, df_dir, image_dir)
    oversampling.img_oversample(savepath_prefix, model_str_list, minority_str_list, image_dir, oversampled_image_dir)
    oversampling.print_messages(savepath_prefix, model_str_list, base_dir)
    oversampling.infovae_reencode(savepath_prefix,
                                  model_str_list,
                                  oversampled_image_dir,
                                  oversampled_vector_dir,
                                  oversampled_dense_vector_dir)


def classify_calibration_train(savepath_prefix, nz, model_str_list, cuda_num, max_iter, key='train'):
    classifier.make_directory(savepath_prefix)

    load_data_dir = os.path.join(savepath_prefix, 'oversampling', 'oversampled-vectors')
    save_dir = os.path.join(savepath_prefix, 'classification')
    
    if key == 'cross-val':
        classifier.cross_val(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir)
    elif key == 'train':
        message_dir = save_dir
        classifier.classifier_train(nz, model_str_list, cuda_num, max_iter, load_data_dir, save_dir, message_dir)



def classify_test(savepath_prefix, nz, model_str_list):
    save_dir = os.path.join(savepath_prefix, 'classification')

    sdss_test_df_path = os.path.join(savepath_prefix, 'latent-vectors', 'sdss', 'test.pkl')
    sdss_test_df = pd.read_pickle(sdss_test_df_path)

    sdss_test_data = sdss_test_df.iloc[:, 0:nz].to_numpy()
    classifier.classifier_test(save_dir, save_dir, model_str_list, sdss_test_data)


def classify_ID_test(savepath_prefix, nz, model_str_list):
    # ID == sim test set
    test_save_dir = os.path.join(savepath_prefix, 'classify-ID')
    for directory in ['prob-df', 'violin-plot']:
        os.makedirs(os.path.join(test_save_dir, directory), exist_ok=True)
    save_dir = os.path.join(savepath_prefix, 'classification')

    dfs = []
    for model_str in model_str_list:
        pkl_path = os.path.join(savepath_prefix, 'latent-vectors', 'test', model_str + '.pkl')
        df = pd.read_pickle(pkl_path)
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=0)
    combined_df.reset_index(drop=True, inplace=True)

    ID_test_data = combined_df.iloc[:, 0:nz].to_numpy()
    classifier.classifier_test(save_dir, test_save_dir, model_str_list, ID_test_data)



if __name__ == "__main__":
    savepath_prefix = 'new-sparse'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)
    

    # infoVAE func
    # img_encoder.vae(savepath_prefix)
    # img_encoder.plot_training(savepath_prefix, 34, 0.0015, 0.0015)
    # img_encoder.plot_residual(savepath_prefix, config, model_str_list, use_cuda=True)
    # img_encoder.encoder(savepath_prefix, model_str_list)
    

    # latent vis func
    # latent_vis.umap_func(savepath_prefix, config['model_params']['latent_dim'], model_str_list)
    # latent_vis.latent_space_vis(savepath_prefix, config, model_str_list, use_cuda=True)


    # minority_str_list = ['AGNrt', 'NOAGNrt', 'TNG50', 'UHDrt', 'n80rt']
    # oversample_sim(savepath_prefix, model_str_list, minority_str_list)


    # cuda_num = str(config['trainer_params']['gpu_id'])
    # max_iter = 300
    # classify_calibration_train(savepath_prefix, config['model_params']['latent_dim'], model_str_list, cuda_num, max_iter, 'cross-val')
    # classify_calibration_train(savepath_prefix, config['model_params']['latent_dim'], model_str_list, cuda_num, max_iter, 'train')


    # classify_test(savepath_prefix, config['model_params']['latent_dim'], model_str_list)


    # classify_ID_test(savepath_prefix, config['model_params']['latent_dim'], model_str_list)


    # classifiers = ['random-forest', 'xgboost', 'stacking-MLP-RF-XGB'] # , 'voting-MLP-RF-XGB']
    sdss_dir = os.path.join(savepath_prefix, 'classification')
    id_dir = os.path.join(savepath_prefix, 'classify-ID')
    # gen = GenOod('stacking-MLP-RF-XGB', savepath_prefix, sdss_dir, id_dir)
    # gen.plot(percent_p=5, x1=-4.92, x2=-4.90, y=150, legend_loc="upper left")
    gen = GenOod('random-forest', savepath_prefix, sdss_dir, id_dir)
    gen.plot(percent_p=5, x1=-4.9245, x2=-4.90, y=110, legend_loc="upper right")
    # gen = GenOod('xgboost', savepath_prefix, sdss_dir, id_dir)
    # gen.plot(percent_p=5, x1=-4.92, x2=-4.90, y=60, legend_loc="upper left")
    # for percent_point in [1, 3, 5, 7, 10, 15]:
    #     gen.select_sdss(percent_point)
    #     gen.print_message(percent_point)
    #     gen.re_classify(model_str_list, config['model_params']['latent_dim'], percent_point)
    # gen.copy_sdss_imgs(percent_point)
    


    # SHAP
    # for c_str in ['random-forest', 'xgboost']:
    #     print(f"SHAP: {c_str}")
    #     xai_func.shap_compute(savepath_prefix,
    #                         os.path.join(savepath_prefix, 'oversampling', 'oversampled-vectors'),
    #                         os.path.join(savepath_prefix, 'gen-ood', 'selected', 'percent5', 'sdss-vectors', c_str, 'id.pkl'),
    #                         config['model_params']['latent_dim'],
    #                         model_str_list,
    #                         c_str)
    #     model_pos_dict = {'AGNrt': 0, 'NOAGNrt': 1, 'TNG100': 2, 'TNG50': 3, 'UHDrt': 4, 'n80rt': 5}
    #     xai_func.shap_plot(savepath_prefix, savepath_prefix, model_pos_dict, c_str, max_display=16)

    # model_pos_dict = {'NOAGNrt': 1}
    # xai_func.shap_plot(savepath_prefix, savepath_prefix, model_pos_dict, 'xgboost', max_display=8)
    
    