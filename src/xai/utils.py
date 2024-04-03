import os


def mkdirs(key):
    dir_list = ['save-shap-values', 'plot']
    for directory in dir_list:
        os.makedirs(os.path.join('src/shap', directory, key), exist_ok=True)


if __name__ == '__main__':
    pass