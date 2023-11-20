import numpy as np
import os


def mkdirs():
    dir_list = ['compare', 'illustris', 'nihao']
    for directory in dir_list:
        os.makedirs(os.path.join('src/shap/save-shap-values', directory), exist_ok=True)

    subdir_list = ['random-forest', 'xgboost', 'single-MLP', 'stacking-MLP-RF-XGB', 'voting-MLP-RF-XGB']
    subsubdir_list = ['global', 'local']
    for directory in dir_list:
        for subdir in subdir_list:
            for subsubdir in subsubdir_list:
                os.makedirs(os.path.join('src/shap/plot', directory, subdir, subsubdir), exist_ok=True)


if __name__ == '__main__':
    mkdirs()