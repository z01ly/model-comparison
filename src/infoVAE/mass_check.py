import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('src/infoVAE/mass_check/mass_SDSS.csv')
    df = df.drop('score', axis=1) # name and mass
    df['name'] = df['name'].astype(str)
    df['m'] = df['m'] / 1e11
    
    image_mass_mapping = dict(zip(df['name'], df['m']))

    
    mass_list = []
    for file_name in os.listdir('src/infoVAE/tsne/selected_sdss'):
        file_name = file_name[: -4]
        if file_name in image_mass_mapping:
            mass_list.append(image_mass_mapping[file_name])
    print(np.array(mass_list).shape)

    range = (0, 4)
    plt.hist(mass_list, bins=10, range=range)
    plt.xlabel('Mass Values (x 10^11)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Selected Mass Values')
    plt.savefig('src/infoVAE/mass_check/selected_mass_hist.png')
    plt.close()

    plt.hist(df['m'], bins=10, color='skyblue', range=range)
    plt.xlabel('Mass Values (x 10^11)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mass Values')
    plt.savefig('src/infoVAE/mass_check/mass_hist.png')
    plt.close()




if __name__ == '__main__':
    main()