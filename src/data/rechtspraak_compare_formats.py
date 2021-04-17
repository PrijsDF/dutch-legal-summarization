import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


def compare_formats(cases_df, year, month, data_dir):
    """ Compares four formats on writing speed and file size. We test CSV, Feather, Parquet with snappy compression,
    and Parquet with brotli compression. The experiment only should be run for one year. The experimental files are
    saved with prefix exp_ to prevent the files from being included in create_final_dataset()
    """
    # Functions used to save the dataframes; these will be called using a dictionary for the four cases
    def save_as_csv(df, path):
        df.to_csv(
            path,
            mode='a',
            index=False,
            header=False
        )

    def save_as_feather(df, path):
        df.to_feather(path)

    def save_as_parquet_snappy(df, path):
        df.to_parquet(path)

    def save_as_parquet_brotli(df, path):
        df.to_parquet(path, compression='brotli')

    logging.info('Starting comparison of file formats...')

    # This df will store the results of the comparison
    comparison_df = pd.DataFrame(data=np.zeros((1, 8)),
                                 columns=['csv_write', 'feather_write', 'snappy_write', 'brotli_write',
                                          'csv_size', 'feather_size', 'snappy_size', 'brotli_size'])

    # A dictionary is used to call the appropiate function depending on the current format
    file_formats = ['csv', 'feather', 'snappy', 'brotli']
    save_format_dict = {
        'csv': save_as_csv,
        'feather': save_as_feather,
        'snappy': save_as_parquet_snappy,
        'brotli': save_as_parquet_brotli
    }

    for file_format in file_formats:
        extension = file_format if file_format == 'csv' or file_format == 'feather' else 'parquet'
        file_path = Path(data_dir / f'temp_cases_content_{year}_{month}_{file_format}.{extension}')

        # Save the df using the current format and compute the time taken
        start_time = time.time()
        save_format_dict[file_format](cases_df, file_path)
        end_time = round(time.time() - start_time, 2)

        # Compute the file_size in MBs
        file_size = round(file_path.stat().st_size / (1024*1024), 2)

        # Save results for current format to results df
        comparison_df.loc[0, ''.join((file_format, '_write'))] = end_time
        comparison_df.loc[0, ''.join((file_format, '_size'))] = file_size

        # Log the results as well
        logging.info((
            f'Writing cases of {year}-{month} to {file_format} took {end_time} seconds, '
            f'file_size: {file_size} bytes'
        ))

        # Finally remove the saved file from disk
        file_path.unlink()

    # Save the df containing this month's results; if its the first month, create the file, otherwise append
    if month == 1:
        comparison_df.to_csv(data_dir / f'{year}_format_comparison.csv', mode='w', index=False)
    else:
        comparison_df.to_csv(data_dir / f'{year}_format_comparison.csv', mode='a', header=False, index=False)

    # print('Finished comparison of file formats.')
    logging.info('Finished comparison of file formats.')