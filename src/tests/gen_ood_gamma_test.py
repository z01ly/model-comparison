import os

import src.config as config
from src.ood import GenOod



if __name__ == "__main__":
    # Previous: gamma = 0.1
    # Test here: gamma = 0.3, 0.5, 0.7, 0.9
    clf = 'xgboost'
    gamma = 0.9
    percent_p = 5

    sdss_dir = config.RESULTS_CLASSIFICATION
    id_dir = config.RESULTS_CLASSIFY_ID
    gen_ood_folder = os.path.join(config.RESULTS_GEN_OOD_GAMMA_TEST, 'gamma' + str(gamma))
    gen = GenOod(clf, sdss_dir, id_dir, savepath=gen_ood_folder, gamma=gamma, M=6)
    gen.plot(percent_p=percent_p, x1=-1.005, x2=-0.99, y=40, legend_loc="upper left")
    
    gen.select_sdss(percent_p)
    gen.print_message(percent_p)
    gen.re_classify(config.MODEL_STR_LIST, config.LATENT_DIM, percent_p)
    # gen.copy_sdss_imgs(percent_p)
