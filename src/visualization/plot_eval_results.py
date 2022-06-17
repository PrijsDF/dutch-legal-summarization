import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# src.utils also loads layout parameters for pyplot
from src.utils import REPORTS_DIR


# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None


def main():
    """In this file, rouge scores will be computed for the generated summaries. We load in a file containing generated
    summaries for either the full dataset or one of the clusters. """
    # Load the file containing the evaluations
    results = pd.read_csv(REPORTS_DIR / 'human_evaluation/evaluation_sample_evaluated.csv')

    # Remove irrelevant columns
    results = results.drop(columns=['Unnamed: 0', 'identifier', 'summary', 'description'
                                    , 'summary_full_model', 'summary_cluster_model'])

    # Print the averages for each of the metrics; make true to look at class-specific results
    get_averages(results, grouped_by_class=True, latex_output=True)

    # Generate a bar graph showing the distribution of the scores for each class
    # plot_score_freqs(results, save_figure=False)


def plot_score_freqs(df, save_figure=False):
    """Choose the type of model for which the data should be shown. Choose from 'true', 'full' and 'cluster' where true
    is the true data. """
    # Pick column names depending on which model needs to be loaded
    names_mapping = {
        'true': ['inf_summary', 'rel_summary', 'flu_summary', 'coh_summary'],
        'full': ['inf_summary_full_model', 'rel_summary_full_model', 'flu_summary_full_model', 'coh_summary_full_model'],
        'cluster': ['inf_summary_cluster_model', 'rel_summary_cluster_model', 'flu_summary_cluster_model', 'coh_summary_cluster_model']
    }

    plot_names_mapping = {
        'true': 'True summaries',
        'full': 'Full model',
        'cluster': 'Cluster model'
    }

    # We will use a figure with three subplots and will go over the plotting process for each model
    models = ['true', 'full', 'cluster']
    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

    for i in range(len(models)):
        col_names = names_mapping[models[i]]

        # Get the frequencies of the scores for each metric
        s1 = df[col_names[0]].astype(int).value_counts()
        s2 = df[col_names[1]].astype(int).value_counts()
        s3 = df[col_names[2]].astype(int).value_counts()
        s4 = df[col_names[3]].astype(int).value_counts()

        # We transpose to get the index, which are the scores, as columns and the metrics as index
        scores_freqs_df = pd.concat([s1, s2, s3, s4], axis=1).transpose()

        category_names = ['1', '2', '3', '4', '5']
        labels = ['Inf.', 'Rel.', 'Flu.', 'Coh.']
        data = scores_freqs_df.to_numpy()
        data = np.nan_to_num(data, nan=0)
        data_cum = data.cumsum(axis=1)
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.05, 0.95, data.shape[1]))

        # Plot for each of the three subplots
        ax[i].invert_yaxis()
        ax[i].set_xlim(0, np.sum(data, axis=1).max())

        for j, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, j]
            starts = data_cum[:, j] - widths
            ax[i].barh(labels, widths, left=starts, height=0.5, label=colname, color=color, zorder=10)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if int(c) > 0:
                    ax[i].text(x, y, str(int(c)), ha='center', va='center', color=text_color, zorder=11)

        # We only want to show the y labels for the first plot so it wont get clutted
        if i != 0:
            ax[i].set_yticks([])

        # Only give the middle plot an x label so it seems as if it is for all three plots
        if i == 1:
            ax[i].set_xlabel('Cumulative frequency')

        ax[i].set_title(plot_names_mapping[models[i]])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 2, 3, 4]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1, loc='lower right')

    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / f'barchart_human_eval.svg', format='svg', dpi=1200)

    plt.show()


def get_averages(df, grouped_by_class, latex_output):
    """Prints the averages of the columns, and thus of each of the metrics that was evaluated. If latex_output is
    True, the output will be printed in a format that can be easily copied into latex. """
    if grouped_by_class:
        # Print the class counts too if necessary
        # print(df.groupby('class').count())

        # We will print the values in such a way that they can immediately be copy pasted into Overleaf
        grouped_means_df = df.groupby(['class']).mean()
        grouped_std_df = df.groupby(['class']).std()

        if latex_output:
            classes = grouped_means_df.index.values
            model_name_mapping = {
                'true': '',
                'full': '_full_model',
                'cluster': '_cluster_model'
            }
            for clas in classes:
                for model_name in model_name_mapping.keys():
                    model_inf = round(grouped_means_df.iloc[clas][f'inf_summary{model_name_mapping[model_name]}'], 2)
                    model_rel = round(grouped_means_df.iloc[clas][f'rel_summary{model_name_mapping[model_name]}'], 2)
                    model_flu = round(grouped_means_df.iloc[clas][f'flu_summary{model_name_mapping[model_name]}'], 2)
                    model_coh = round(grouped_means_df.iloc[clas][f'coh_summary{model_name_mapping[model_name]}'], 2)

                    model_inf_std = round(grouped_std_df.iloc[clas][f'inf_summary{model_name_mapping[model_name]}'], 2)
                    model_rel_std = round(grouped_std_df.iloc[clas][f'rel_summary{model_name_mapping[model_name]}'], 2)
                    model_flu_std = round(grouped_std_df.iloc[clas][f'flu_summary{model_name_mapping[model_name]}'], 2)
                    model_coh_std = round(grouped_std_df.iloc[clas][f'coh_summary{model_name_mapping[model_name]}'], 2)
                    print(f'Class {clas} {model_name} & {model_inf}\\textpm{model_inf_std} '
                          f'& {model_rel}\\textpm{model_rel_std} '
                          f'& {model_flu}\\textpm{model_flu_std} '
                          f'& {model_coh}\\textpm{model_coh_std} \\\\')
        else:
            # Just print the means and stds tables normally
            print(grouped_means_df)
            print(grouped_std_df)

    else:  # in this case we just want the overall averages
        agg_df = pd.concat([df.mean(), df.std()], axis=1).rename(columns={0: 'Mean', 1: 'Std'})

        print(agg_df)


if __name__ == '__main__':
    main()
