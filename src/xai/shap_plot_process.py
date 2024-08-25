import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import easyocr
import shap
from PIL import Image, ImageOps


def local_plot():
    pass


def global_plot(prefix, classifier_key, model_str, model_pos, max_display=16):
    clf = pickle.load(open(os.path.join(prefix, 'classification', 'save-model', classifier_key + '-model.pickle'), "rb"))
    # print(clf.classes_)

    with open(os.path.join(prefix, 'xai', 'shap', 'save-shap-values', classifier_key + '.sav'), 'rb') as f:
        shap_values = pickle.load(f)
    print(shap_values.shape)

    fig = plt.figure()

    shap.plots.beeswarm(shap_values[:, :, model_pos], max_display=max_display, show=False)
    # shap.plots.beeswarm(shap_values, max_display=20, show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(prefix, 'xai', 'shap', 'beeswarm-plot', classifier_key, model_str + '.png'))
    plt.close()



def ocr(shap_plot_path, temporary_path):
    image = Image.open(shap_plot_path)
    width, height = image.size # (800, 950)
    # print(image.size)
    image_part = image.crop((0, 0, width // 2 - 120, height - 90)) # Feature text part is left
    image_part.save(temporary_path)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(temporary_path)

    text_list = [detection[1] for detection in result]
    # print(f"text list: \n {text_list} \n")
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
        print(f'The temporary file {temporary_path} has been deleted.')
    else:
        print(f'The file {temporary_path} does not exist.')

    feature_list = [item for item in text_list if item.startswith('Feature')]
    print(f"feature list length: {len(feature_list)}")
    print(f"feature list: \n {feature_list}")
    
    return feature_list



def stack_pngs(feature_list, source_dir, output_dir, output_name, shap_plot_path, tempo_stack):
    file_list = [f'dim{item.split()[-1]}.png' for item in feature_list]

    resized_width, resized_height = 1440, 180
    resized_images = []
    for file_name in file_list:
        image_path = f'{source_dir}/{file_name}'
        img = Image.open(image_path) # size (7500, 900)
        img = img.resize((resized_width, resized_height))
        resized_images.append(img)

    stacked_image = Image.new('RGB', (resized_width, resized_height * len(file_list)))

    for i, img in enumerate(resized_images):
        stacked_image.paste(img, (0, i * resized_height))

    white_spaces = Image.new('RGB', (stacked_image.size[0], stacked_image.size[1] + 600), color='white')
    white_spaces.paste(stacked_image, (0, 200))

    tempo_stack_path = f'{output_dir}/{tempo_stack}'
    white_spaces.save(tempo_stack_path)

    # Open the images
    stacked_image = Image.open(tempo_stack_path)
    beeswarm_image = Image.open(shap_plot_path)
    # print(beeswarm_image.size)

    # Ensure both images have the same height
    max_height = max(stacked_image.size[1], beeswarm_image.size[1])
    stacked_image = stacked_image.resize((stacked_image.size[0], max_height))
    beeswarm_image = beeswarm_image.resize((beeswarm_image.size[0] + 2000, max_height))

    # Calculate the width of the new image
    new_width = stacked_image.size[0] + beeswarm_image.size[0]

    # Create a new image with the calculated width and the maximum height of the two images
    combined_image = Image.new('RGB', (new_width, max_height))

    # Paste the stacked image on the left and the beeswarm image on the right
    combined_image.paste(stacked_image, (0, 0))
    combined_image.paste(beeswarm_image, (stacked_image.size[0], 0))

    # Save the combined image
    combined_image.save(f'{output_dir}/{output_name}')

    if os.path.exists(tempo_stack_path):
        os.remove(tempo_stack_path)
        print(f'The temporary file {tempo_stack_path} has been deleted. \n')
    else:
        print(f'The file {tempo_stack_path} does not exist. \n')



def process_main(prefix, savepath_prefix, model_str_dict, classifier_key):
    # OCR accuracy is not 100% and requires manual check
    for model_str, model_pos in model_str_dict.items():
        global_plot(prefix, classifier_key, model_str, model_pos)

        shap_plot_path = os.path.join(prefix, 'xai', 'shap', 'beeswarm-plot', classifier_key, model_str + '.png')
        temporary_path = os.path.join(prefix, 'xai', 'shap', 'beeswarm-plot', 'left_half.png')
        feature_list = ocr(shap_plot_path, temporary_path)

        source_dir = os.path.join(savepath_prefix, 'vis', 'latent-space', 'dim-example', model_str)
        output_dir = os.path.join(prefix, 'xai', 'shap', 'beeswarm-plot', classifier_key) 
        stack_pngs(feature_list, source_dir, output_dir, model_str+'-stack.png', shap_plot_path, 'tempo_image.png')



if __name__ == "__main__":
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    # model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    model_str_dict = {'NOAGNrt': 1, 'TNG100': 2} # model for plot

    process_main(savepath_prefix, nz, model_str_dict, 'random-forest')

