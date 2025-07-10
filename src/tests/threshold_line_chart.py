import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import src.config as config


def percent_line_chart():
    # Data for thresholds and models
    thresholds = [99, 97, 95, 93, 90, 85]
    rf_id = [83, 62, 42, 29, 13, 5]
    xgb_id = [87, 73, 71, 68, 63, 54]
    stacking_id = [77, 61, 55, 47, 35, 25]

    rf_ood = [17, 38, 58, 71, 87, 95]
    xgb_ood = [13, 27, 29, 32, 37, 46]
    stacking_ood = [23, 39, 45, 53, 65, 75]

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, rf_id, color='blue', marker='o', label='Random Forest')
    plt.plot(thresholds, xgb_id, color='green', marker='o', label='XGBoost')
    plt.plot(thresholds, stacking_id, color='red', marker='o', label='Stacking-MLP-RF-XGB')
    plt.title('ID Ratio vs Threshold')
    plt.xlabel('Threshold (%)')
    plt.ylabel('ID Ratio (%)')

    # Annotate the points with the percentage values
    for i in range(len(thresholds)):
        plt.text(thresholds[i], rf_id[i] + 2, f'{rf_id[i]}%', ha='center')
        plt.text(thresholds[i], xgb_id[i] + 2, f'{xgb_id[i]}%', ha='center')
        plt.text(thresholds[i], stacking_id[i] + 2, f'{stacking_id[i]}%', ha='center')

    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_GEN_OOD, 'plot', 'id_ratio_threshold.png'))  
    plt.close() 

    # Plotting OOD ratio and saving it as a separate figure
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, rf_ood, color='blue', marker='o', label='Random Forest')
    plt.plot(thresholds, xgb_ood, color='green', marker='o', label='XGBoost')
    plt.plot(thresholds, stacking_ood, color='red', marker='o', label='Stacking-MLP-RF-XGB')
    plt.title('OOD Ratio vs Threshold')
    plt.xlabel('Threshold (%)')
    plt.ylabel('OOD Ratio (%)')

    # Annotate the points with the percentage values
    for i in range(len(thresholds)):
        plt.text(thresholds[i], rf_ood[i] + 2, f'{rf_ood[i]}%', ha='center')
        plt.text(thresholds[i], xgb_ood[i] + 2, f'{xgb_ood[i]}%', ha='center')
        plt.text(thresholds[i], stacking_ood[i] + 2, f'{stacking_ood[i]}%', ha='center')

    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_GEN_OOD, 'plot', 'ood_ratio_threshold.png'))  
    plt.close() 



if __name__ == "__main__":
    # Plot the figure without grid
    percent_line_chart()