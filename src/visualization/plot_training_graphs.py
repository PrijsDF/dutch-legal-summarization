import pandas as pd
import matplotlib.pyplot as plt

# src.utils also loads layout parameters for pyplot
from src.utils import COLORS, REPORTS_DIR


# These two vars are used to style some of the plots
cluster_alpha = 0.7  # The opacity of cluster model lines; to make the main model line better visible
linestyle = 'dashed'  # The type of line for the validation curves in the val loss all models graph


def main(graphs_dir=REPORTS_DIR / 'training_graphs'):
    """Create a plot of the training graphs. The source of the data is tensorboard; during training it logged training
    and evaluation losses. In tensorboard, csv files containing these graphs could be downloaded, as I did. The plots
    that are created with this file, will be presented in the thesis.

    If more than one of the four plots is drawn, all but the first drawn plot will have the wrong size (not 14,7)
    """

    # Pretraining - Generate the plot containing both of the loss graphs
    pt_create_plot(fine_tune_folder=graphs_dir, save_figure=False)

    # Finetuning - Generate train and val plot for the aggregate cluster model and the full model
    ft_create_plot_full_models(fine_tune_folder=graphs_dir, save_figure=False)

    # Finetuning - Generate train and val plot containg the loss for each of the models individually
    ft_create_plot_all_models(fine_tune_folder=graphs_dir, x_axis='index', save_figure=False)

    # Graph that compares full model with a 10 epoch model to show overfitting
    ft_43e_plot(fine_tune_folder=graphs_dir, save_figure=False)


def pt_create_plot(fine_tune_folder, save_figure=False):
    """Create a figure showing the training and validation loss of the pretraining phase of BART."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    pt_train_file_path = fine_tune_folder / 'pretraining/run-pt_65p_2-tag-train_loss.csv'
    pt_val_file_path = fine_tune_folder / 'pretraining/run-pt_65p_2-tag-eval_loss.csv'

    cols = ['Step', 'Value']
    train_df = pd.read_csv(pt_train_file_path, usecols=cols)
    val_df = pd.read_csv(pt_val_file_path, usecols=cols)

    # With val column names prefixed with val_, we don't struggle with joining the dataframes later
    val_df = val_df.rename(columns={'Step': 'Val_Step', 'Value': 'Val_Value'})

    # Plot the data
    plt.plot(train_df['Step'], train_df['Value'], label="train")
    plt.plot(val_df['Val_Step'], val_df['Val_Value'], label="validation")

    # To highlight the strange disturbance in the losses
    vertical_lines = [
        [5000, '5000'],
        [35000, '35000'],
        [82500, '82500']
    ]
    for line in vertical_lines:
        x_coord = line[0]
        label = line[1]

        plt.axvline(x=x_coord, color='black', linestyle='--', zorder=-10, linewidth=1)
        plt.annotate(label, (x_coord+2000, 0.1), fontsize=13)

    plt.xlabel('Step')
    plt.ylabel('Cross-entropy Loss')
    x_max = train_df['Step'].max()
    y_max = train_df['Value'].max()
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()

    plt.grid()
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / 'pt_losses_graph_65p.svg', format='svg', dpi=1200)

    plt.show()

    return True


def ft_create_plot_full_models(fine_tune_folder, save_figure=False):
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
    plt.plot(full_train_df.index.values, full_train_df['Value'], COLORS['blue'], label="Full model")
    plt.plot(full_train_df.index.values, weighted_train_loss, COLORS['red'], label="Cluster model")
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
    plt.plot(x_axis_vals, full_val_df['Value'], COLORS['blue'], label="Full model")
    plt.plot(x_axis_vals, weighted_val_loss, COLORS['red'], label="Cluster model")
    x_max = max(x_axis_vals)
    plt.xlabel('Epoch')
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()

    plt.grid()
    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / f'ft_losses_graph_full_models.svg', format='svg', dpi=1200)

    plt.show()

    return True


def ft_create_plot_all_models(fine_tune_folder, x_axis='index', save_figure=False):
    """Create a figure showing the fine-tuning plots for each of the individual cluster models and the model that uses
    all data. Both the train and validation losses are shown."""
    # We will load both the training and the validation loss files and merge both together in 1 dataframe
    cols = ['Step', 'Value']

    # These two mappings are used to pick the right file and color
    file_mapping = {
        'Full model': 'ft_full',
        'Cluster 0 model': 'ft_cluster_0',
        'Cluster 1 model': 'ft_cluster_1',
        'Cluster 2 model': 'ft_cluster_2',
        'Cluster 3 model': 'ft_cluster_3',
        'Cluster 4 model': 'ft_cluster_4',
        'Cluster 5 model': 'ft_cluster_5'
    }

    color_mapping = {
        'Full model': COLORS['blue'],
        'Cluster 0 model': COLORS['pink'],
        'Cluster 1 model': COLORS['purple'],
        'Cluster 2 model': COLORS['orange'],
        'Cluster 3 model': COLORS['yellow'],
        'Cluster 4 model': COLORS['green'],
        'Cluster 5 model': COLORS['brown']
    }

    # Now we will start plotting; to do this we loop over the files and plot them one by one
    # First for the train losses
    plt.subplot(1, 2, 1)
    plt.title('Training Loss')
    x_max = 0  # This needs to have the value of the full model max
    for model_name, file_id in file_mapping.items():
        # Load the results of the model
        train_results = pd.read_csv(fine_tune_folder / f'run-{file_id}-tag-train_loss.csv', usecols=cols)

        # Pick the x_vals depending on the x_axis param
        # Furthermore, we need the x_max to be set with the full model max
        if x_axis == 'step':
            if model_name == 'Full model':
                x_max = train_results['Step'].max()
                plt.plot(train_results['Step'], train_results['Value'], color_mapping[model_name],
                         label=model_name, zorder=10)
            else:
                plt.plot(train_results['Step'], train_results['Value'], color_mapping[model_name],
                         label=model_name, alpha=cluster_alpha, linestyle=linestyle)
        else:
            if model_name == 'Full model':
                x_max = train_results.index.values.max()
                plt.plot(train_results.index.values, train_results['Value'], color_mapping[model_name],
                         label=model_name, zorder=10)
            else:
                plt.plot(train_results.index.values, train_results['Value'], color_mapping[model_name],
                         label=model_name, alpha=cluster_alpha, linestyle=linestyle)

    x_label = 'Step' if x_axis == 'index' else 'Epoch'
    plt.xlabel(x_label)
    plt.ylabel('Cross-entropy Loss')
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()
    plt.grid()

    # Now, draw the right plot, containing the val losses
    plt.subplot(1, 2, 2)
    plt.title('Validation Loss')
    x_max = 0  # This needs to have the value of the full model max
    for model_name, file_id in file_mapping.items():
        # Load the results of the model
        val_results = pd.read_csv(fine_tune_folder / f'run-{file_id}-tag-eval_loss.csv', usecols=cols)

        # Pick the x_vals depending on the x_axis param
        # Furthermore, we need the x_max to be set with the full model max
        if x_axis == 'step':
            if model_name == 'Full model':
                x_max = val_results['Step'].max()
                plt.plot(val_results['Step'], val_results['Value'], color_mapping[model_name],
                         label=model_name, zorder=10)
            else:
                plt.plot(val_results['Step'], val_results['Value'], color_mapping[model_name],
                         label=model_name, alpha=cluster_alpha, linestyle=linestyle)
        else:
            if model_name == 'Full model':
                x_max = val_results.index.values.max()
                plt.plot(val_results.index.values, val_results['Value'], color_mapping[model_name],
                         label=model_name, zorder=10)
            else:
                plt.plot(val_results.index.values, val_results['Value'], color_mapping[model_name],
                         label=model_name, alpha=cluster_alpha, linestyle=linestyle)

    x_label = 'Step' if x_axis == 'index' else 'Epoch'
    plt.xlabel(x_label)
    plt.ylabel('Cross-entropy Loss')
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)

    plt.legend()
    plt.grid()

    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / f'ft_losses_graph_all_models_{x_axis}.svg', format='svg', dpi=1200)

    plt.show()

    return True


def ft_43e_plot(fine_tune_folder, save_figure=False):
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
    plt.plot(full_train_df['Step'], full_train_df['Value'], COLORS['blue'], label="Full model")
    plt.plot(class_0_train_df['Step'], class_0_train_df['Value'], COLORS['pink'], linestyle=linestyle, label="Cluster 0 model")
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
    plt.plot(full_val_df['Step'], full_val_df['Value'], COLORS['blue'], label="Full model")
    plt.plot(class_0_val_df['Step'], class_0_val_df['Value'], COLORS['pink'], linestyle=linestyle, label="Cluster 0 model")
    plt.xlabel('Step')
    x_max = full_train_df['Step'].max()
    # full_df['Value'].max() # If we do .max() the graph will be too high, which defeats its purpose
    y_max = 1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.legend()
    plt.grid()

    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / 'ft_losses_graph_43e_models.svg', format='svg', dpi=1200)
    plt.show()

    return True


if __name__ == '__main__':
    main()
