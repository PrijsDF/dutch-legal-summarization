import time
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset

blue = '#1F77B4'  # Use solely for the full model
red = '#ff0000'  # Use solely for the cluster combined model

geel = '#ff00f0'
marine = '#9400d3' #'#35193E'#141E8C'
olive = '#ff4500' #'#AD1759' #'#808000'
purple = '#ffa500' #'#F37651' #'#2A0800'
grass = '#009900' #'#E13342' #'#28b463'
pink = '#800000' #'#701F57'  #'#F6B48F' #'#b428a7'

cluster_alpha = 0.7  # The opacity of cluster model lines; to make hte main model line better visible
linestyle = 'dashed'  # The type of line for the validation curves in the val loss all models graph

# Font sizes for graph texts
small_size = 16
medium_size = 20
big_size = 16

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=small_size)  # controls default text sizes
plt.rc('text', usetex=True)
plt.figure(figsize=(14, 7))#, dpi=160)  #, dpi=300)
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=big_size)     # fontsize of the figure title


def main(graphs_dir=REPORTS_DIR / 'training_graphs'):
    """Create a plot of the training graphs. The source of the data is tensorboard; during training it logged training
    and evaluation losses. In tensorboard, csv files containing these graphs could be downloaded, as I did. The plots
    that are created with this file, will be presented in the thesis."""

    # Pretraining - Generate the plot containing both of the loss graphs
    pt_train_file_path = graphs_dir / 'pretraining/run-pt_65p_2-tag-train_loss.csv'
    pt_val_file_path = graphs_dir / 'pretraining/run-pt_65p_2-tag-eval_loss.csv'
    # pt_create_plot(pt_train_file_path, pt_val_file_path)

    # Finetuning - Generate train and val plot for the aggregate cluster model and the full model
    # ft_create_plot_full_models(fine_tune_folder=graphs_dir, x_axis='index')

    # Finetuning - Generate train and val plot containg the loss for each of the models individually
    # ft_create_plot_all_models(fine_tune_folder=graphs_dir, x_axis='step')

    # Graph that compares full model with a 10 epoch model to show overfitting
    # ft_43e_plot(fine_tune_folder=graphs_dir)


def pt_create_plot(train_path, val_path):
    """Create a figure showing the training and validation loss of the pretraining phase of BART."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    cols = ['Step', 'Value']
    train_df = pd.read_csv(train_path, usecols=cols)
    val_df = pd.read_csv(val_path, usecols=cols)

    # With val column names prefixed with val_, we don't struggle with joining the dataframes later
    val_df = val_df.rename(columns={'Step': 'Val_Step', 'Value': 'Val_Value'})

    # Plot the data
    # We use below layout; this layout is also used in other files
    #plt.style.use('seaborn-whitegrid')  # ggplot - seaborn - searborn-darkgrid
    #plt.suptitle('Loss During BART Pretraining')
    #plt.subplots_adjust(left=0.055, bottom=0.06, right=0.98, top=0.9, wspace=0.1, hspace=None)

    plt.plot(train_df['Step'], train_df['Value'], label="train")
    plt.plot(val_df['Val_Step'], val_df['Val_Value'], label="validation")
    plt.xlabel('Step')
    plt.ylabel('Cross-entropy Loss')
    x_max = train_df['Step'].max()
    y_max = train_df['Value'].max()
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()

    plt.grid()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'pt_losses_graph_65p.svg', format='svg', dpi=1200)
    plt.show()

    return True


def ft_create_plot_full_models(fine_tune_folder, x_axis='index'):
    """Create a figure showing the fine-tuning plots for the combined clustered models and the model that uses
    all data. In this figure we thus take a weighted average of the losses of the cluster models and compare the
    obtained model with the total model."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    cols = ['Step', 'Value']

    # Load the result csv's
    full_train_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-train_loss.csv', usecols=cols)
    full_val_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-eval_loss.csv', usecols=cols)

    # For each of the CSVs, we add a weighted value which corresponds to (cases_in_cluster/total_cases) * loss_value
    # The lengths of all data splits were computed in the file find_dataset_distributions.py
    # These mappings map the number of cases of the train split of each cluster dataset to its name
    train_mapping = {
        '0': 16509,
        '1': 9453,
        '2': 13409,
        '3': 4668,
        '4': 10789,
        '5': 15312,
    }

    val_mapping = {
        '0': 4740,
        '1': 2714,
        '2': 3850,
        '3': 1341,
        '4': 3098,
        '5': 4397,
    }

    # These total values are used to weigh the individual scores
    train_size = sum([value for key, value in train_mapping.items()])
    val_size = sum([value for key, value in val_mapping.items()])

    # weighted_loss vars hold the weighted loss for the cluster models for each timestep; this is the list that will be
    # plotted. The train loss var has 11 values (10 epochs + step 1) whereas the val loss var has only 10
    weighted_train_loss = [0 for i in range(len(full_train_df))]
    for key, value in train_mapping.items():
        cluster = key
        cluster_weight = value / train_size
        cluster_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_{cluster}-tag-train_loss.csv', usecols=cols)

        # Weigh each of the values for this class and store them in the weighted loss var
        for i in range(len(weighted_train_loss)):
            loss = cluster_train_df.iloc[i]['Value']
            weighted_loss = loss * cluster_weight
            weighted_train_loss[i] = weighted_train_loss[i] + weighted_loss

    # Now the same for the val losses
    weighted_val_loss = [0 for i in range(len(full_val_df))]
    for key, value in val_mapping.items():
        cluster = key
        cluster_weight = value / val_size
        cluster_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_{cluster}-tag-eval_loss.csv', usecols=cols)

        # Weigh each of the values for this class and store them in the weighted loss var
        for i in range(len(weighted_val_loss)):
            loss = cluster_val_df.iloc[i]['Value']
            weighted_loss = loss * cluster_weight
            weighted_val_loss[i] = weighted_val_loss[i] + weighted_loss

    # Draw the left plot, containing the train losses
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.plot(full_train_df.index.values, full_train_df['Value'], blue, label="Full model")
    plt.plot(full_train_df.index.values, weighted_train_loss, red, label="Cluster model")
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy Loss')
    x_max = full_train_df.index.values.max()
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()
    plt.grid()

    # Draw the right plot, containing the val losses
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss')
    # We need this adjusted axis because val starts at 1, not at 0
    x_axis_vals = [i for i in range(1, 11)]
    plt.plot(x_axis_vals, full_val_df['Value'], blue, label="Full model")
    plt.plot(x_axis_vals, weighted_val_loss, red, label="Cluster model")
    #plt.ylabel('Cross-entropy Loss')
    x_max = max(x_axis_vals)
    plt.xlabel('Epoch')
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()

    plt.grid()
    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f'ft_losses_graph_full_models.svg', format='svg', dpi=1200)
    plt.show()

    return True


def ft_create_plot_all_models(fine_tune_folder, x_axis='index'):
    """Create a figure showing the fine-tuning plots for each of the individual cluster models and the model that uses
    all data.."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    cols = ['Step', 'Value']

    # Load the result csv's
    full_train_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-train_loss.csv', usecols=cols)
    class_0_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_0-tag-train_loss.csv', usecols=cols)
    class_1_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_1-tag-train_loss.csv', usecols=cols)
    class_2_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_2-tag-train_loss.csv', usecols=cols)
    class_3_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_3-tag-train_loss.csv', usecols=cols)
    class_4_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_4-tag-train_loss.csv', usecols=cols)
    class_5_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_5-tag-train_loss.csv', usecols=cols)

    full_val_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-eval_loss.csv', usecols=cols)
    class_0_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_0-tag-eval_loss.csv', usecols=cols)
    class_1_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_1-tag-eval_loss.csv', usecols=cols)
    class_2_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_2-tag-eval_loss.csv', usecols=cols)
    class_3_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_3-tag-eval_loss.csv', usecols=cols)
    class_4_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_4-tag-eval_loss.csv', usecols=cols)
    class_5_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_5-tag-eval_loss.csv', usecols=cols)

    # Draw the left plot, containing the train losses
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    if x_axis == 'step':
        plt.plot(full_train_df['Step'], full_train_df['Value'], blue, label="Full model", zorder=10)
        plt.plot(class_0_train_df['Step'], class_0_train_df['Value'], geel, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 0 model")
        plt.plot(class_1_train_df['Step'], class_1_train_df['Value'], marine, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 1 model")
        plt.plot(class_2_train_df['Step'], class_2_train_df['Value'], olive, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 2 model")
        plt.plot(class_3_train_df['Step'], class_3_train_df['Value'], purple, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 3 model")
        plt.plot(class_4_train_df['Step'], class_4_train_df['Value'], grass, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 4 model")
        plt.plot(class_5_train_df['Step'], class_5_train_df['Value'], pink, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 5 model")
        plt.xlabel('Step')
        plt.ylabel('Cross-entropy Loss')
        x_max = full_train_df['Step'].max()
    elif x_axis == 'index':
        plt.plot(full_train_df.index.values, full_train_df['Value'], blue, label="Full model", zorder=10)
        plt.plot(class_0_train_df.index.values, class_0_train_df['Value'], geel, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 0 model")
        plt.plot(class_1_train_df.index.values, class_1_train_df['Value'], marine, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 1 model")
        plt.plot(class_2_train_df.index.values, class_2_train_df['Value'], olive, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 2 model")
        plt.plot(class_3_train_df.index.values, class_3_train_df['Value'], purple, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 3 model")
        plt.plot(class_4_train_df.index.values, class_4_train_df['Value'], grass, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 4 model")
        plt.plot(class_5_train_df.index.values, class_5_train_df['Value'], pink, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 5 model")
        plt.xlabel('Epoch')
        plt.ylabel('Cross-entropy Loss')
        x_max = full_train_df.index.values.max()
    # full_df['Value'].max() # If we do .max() the graph will be too high, which defeats its purpose
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()
    plt.grid()

    # Draw the right plot, containing the val losses
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss')
    if x_axis == 'step':
        plt.plot(full_val_df['Step'], full_val_df['Value'], blue, label="Full model", zorder=10)
        plt.plot(class_0_val_df['Step'], class_0_val_df['Value'], geel, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 0 model")
        plt.plot(class_1_val_df['Step'], class_1_val_df['Value'], marine, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 1 model")
        plt.plot(class_2_val_df['Step'], class_2_val_df['Value'], olive, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 2 model")
        plt.plot(class_3_val_df['Step'], class_3_val_df['Value'], purple, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 3 model")
        plt.plot(class_4_val_df['Step'], class_4_val_df['Value'], grass, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 4 model")
        plt.plot(class_5_val_df['Step'], class_5_val_df['Value'], pink, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 5 model")
        plt.xlabel('Step')
        # plt.ylabel('Cross-entropy Loss')
        x_max = full_val_df['Step'].max()
    elif x_axis == 'index':
        # If we just use e.g. full_train_df.index.values, we will have an x-axis from 0 to 9
        # This is only relevant for the Val loss however (and not for the train loss)!
        x_axis_vals = [i for i in range(1, 11)]
        plt.plot(x_axis_vals, full_val_df['Value'], blue, label="Full model", zorder=10)
        plt.plot(x_axis_vals, class_0_val_df['Value'], geel, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 0 model")
        plt.plot(x_axis_vals, class_1_val_df['Value'], marine, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 1 model")
        plt.plot(x_axis_vals, class_2_val_df['Value'], olive, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 2 model")
        plt.plot(x_axis_vals, class_3_val_df['Value'], purple, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 3 model")
        plt.plot(x_axis_vals, class_4_val_df['Value'], grass, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 4 model")
        plt.plot(x_axis_vals, class_5_val_df['Value'], pink, alpha=cluster_alpha, linestyle=linestyle, label="Cluster 5 model")
        plt.xlabel('Epoch')
        # plt.ylabel('Cross-entropy Loss')
        x_max = max(x_axis_vals)
    # full_df['Value'].max() # If we do .max() the graph will be too high, which defeats its purpose
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()

    plt.grid()
    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f'ft_losses_graph_all_models_{x_axis}.svg', format='svg', dpi=1200)
    plt.show()

    return True


def ft_43e_plot(fine_tune_folder):
    """This function draws the figure that shows the 43 epoch cluster 0 model versus the full data model. Reason for
    this comparison, is to see what happens if we have one of the clustering models train for as many steps as the full
    model (results in overfitting)."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    cols = ['Step', 'Value']

    # Load the result csv's
    full_train_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-train_loss.csv', usecols=cols)
    full_val_df = pd.read_csv(fine_tune_folder / f'run-ft_full-tag-eval_loss.csv', usecols=cols)
    class_0_train_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_0_43e-tag-train_loss.csv', usecols=cols)
    class_0_val_df = pd.read_csv(fine_tune_folder / f'run-ft_cluster_0_43e-tag-eval_loss.csv', usecols=cols)

    # Draw the left plot, containing the train losses
    plt.xlabel('Step')
    plt.ylabel('Cross-entropy Loss')
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    plt.plot(full_train_df['Step'], full_train_df['Value'], blue, label="Full model")
    plt.plot(class_0_train_df['Step'], class_0_train_df['Value'], geel, linestyle=linestyle, label="Cluster 0 model")
    plt.xlabel('Step')
    plt.ylabel('Cross-entropy Loss')
    x_max = full_train_df['Step'].max()
    # full_df['Value'].max() # If we do .max() the graph will be too high, which defeats its purpose
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.legend()
    plt.grid()

    # Draw the left plot, containing the validation losses
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss')
    plt.plot(full_val_df['Step'], full_val_df['Value'], blue, label="Full model")
    plt.plot(class_0_val_df['Step'], class_0_val_df['Value'], geel, linestyle=linestyle, label="Cluster 0 model")
    plt.xlabel('Step')
    #plt.ylabel('Cross-entropy Loss')
    x_max = full_train_df['Step'].max()
    # full_df['Value'].max() # If we do .max() the graph will be too high, which defeats its purpose
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.legend()

    plt.grid()
    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'ft_losses_graph_43e_models.svg', format='svg', dpi=1200)
    plt.show()

    return True


if __name__ == '__main__':
    main()
