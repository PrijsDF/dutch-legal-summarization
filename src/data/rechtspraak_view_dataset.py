import pandas as pd
from src.utils import DATA_DIR

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None


def main():
    """View Open Rechtspraak dataset with pandas."""
    dataset_dir = DATA_DIR / 'raw/OpenDataUitspraken'
    complete_dataset_path = dataset_dir / 'cases_content.parquet'

    # Read the parquet containing the df into a pd df
    # cases_content = pd.read_parquet(complete_dataset_path)

    # Read multiple parquet files into a df
    cases_content = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in dataset_dir.glob('*_198*.parquet')
    )

    # Get a sample of the dataset and save the sample as csv
    samples_df = create_sample_of_df(cases_content, number_of_items=20, only_complete_items=True,
                                     save_sample=True, save_dir=dataset_dir)

    # View a specific ECLI if needed
    # print(cases_content.loc[cases_content['identifier'] == 'ECLI:NL:ORBBNAA:1994:BU4842'])

    # Or a single string value (.values[0] is needed for this)
    # print(cases_content.loc[cases_content['identifier'] == 'ECLI:NL:ORBBNAA:1994:BU4842', 'description'].values[0])

    # View the sample
    print(samples_df)
    print(samples_df.dtypes)

    # If needed, save the dataset to csv for later inspection
    cases_content.to_csv(dataset_dir / 'cases_content.csv', mode='w', index=False, header=True)


def create_sample_of_df(df, number_of_items=20, only_complete_items=True, save_sample=False, save_dir=None):
    """ Returns a subset of the df containing number_of_items cases. By default, only complete cases are included (these
    are cases with both a summary and a description). Furthermore, if needed the sample can be saved."""
    # Subset the df to only include complete cases
    if only_complete_items:
        df = df.loc[df['missing_parts'] == 'none']

    # Pick sample
    samples_df = df.sample(n=number_of_items, random_state=1)

    if save_sample:
        samples_df.to_csv(save_dir / 'sample_cases_content.csv', mode='w', index=False, header=True)

    return samples_df


if __name__ == '__main__':
    main()
