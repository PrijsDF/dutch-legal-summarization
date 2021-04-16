import logging
import time
from pathlib import Path

from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, LOG_DIR
from rechtspraak_parse_xml_functions import parse_xml


def main():
    """ Runs data processing scripts to turn external data from (../external) into
        a raw dataset (saved in ../raw) that can be used as a starting point for creating the
        final dataset.
    """
    # Start logging
    print('Making raw dataset from external Rechtspraak data...')
    logging.info('Making raw dataset from external Rechtspraak data...')

    # We will store the intermediary files and final raw dataset here
    cases_dir = DATA_DIR / 'raw/OpenDataUitspraken'

    # Specify years; 1912 does not exist
    years = list(range(1911, 2022))
    years.remove(1912)
    years = range(1994, 1995)  # Temp

    # Get month archives of each year; we will loop over these
    all_month_archives = []
    for year in years:
        year_path = DATA_DIR / f'external/OpenDataUitspraken/{str(year)}'
        month_archives = [x for x in year_path.iterdir() if x.is_file()]
        all_month_archives += month_archives

    # Start processing all archives
    pbar = tqdm(all_month_archives)
    for month_archive in pbar:
        # To compute time per month archive
        start = time.time()

        # Used when saving files later on and for configuration of the progress bar
        archive_name = month_archive.stem

        # Config progress bar
        year = archive_name[:4]
        month = archive_name[4:]
        pbar.set_description(f'Processing month {month} of {year}')

        # Parse all cases in month archive
        cases_content_df = process_month_archive(month_archive)

        # Save the extracted archive's cases to a parquet
        cases_content_df.to_parquet(cases_dir / f'cases_content_{archive_name}.parquet')

        # # Uncomment if conducting comparison experiment for the different file formats
        # compare_formats(cases_content_df, cases_dir, archive_name)

        logging.info(f'{archive_name} has been parsed and saved. Time taken: {time.time() - start}')

    print('Finished parsing archives.')
    logging.info('All month archives have been parsed and saved.')

    # Combine the individual snappy parquet files into the final dataset file using brotli compression
    create_final_dataset(cases_dir)


def process_month_archive(month_archive):
    """ Processes each XML file (i.e. legal case) in the month archive, returning a df with the content of these cases.
    """
    # Store all cases of current archive in these dfs
    cases_content_df = pd.DataFrame(columns=[
        'identifier', 'missing_parts', 'case_type', 'case_number', 'jurisdiction', 'creator', 'judgment_date',
        'relation', 'procedures', 'seat_location', 'references', 'publisher', 'issue_date', 'modified',
        'summary', 'description',  # 'language', # 'format'
    ])

    # Make archive containing all the legal cases of the month's zip file
    archive = ZipFile(month_archive, 'r')
    cases_archive = archive.namelist()

    for legal_case in cases_archive:
        # Read the content of the zip file (XML) into bf4 parser
        case_content = parse_xml(archive.read(legal_case))

        # Append to df
        cases_content_df = cases_content_df.append(case_content, ignore_index=True)

    return cases_content_df


def create_final_dataset(data_dir):
    """ Combines all individual snappy parquet files into one complete dataset. This dataset then is saved as
    a parquet using brotli compression, reducing file size as much as possible.
    """
    start = time.time()
    print('Merging the individual parquet files...')
    logging.info('Merging the individual parquet files...')

    # Below, we combine the individual snappy parquet files into the final dataset file using brotli compression
    all_parquets = list(data_dir.glob('cases_content_*.parquet'))

    # Add parquet files to one df (code from https://stackoverflow.com/a/52193992)
    complete_df = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in all_parquets
    )

    # Save the new complete df
    complete_df.to_parquet(data_dir / 'cases_content.parquet', compression='brotli')

    # Delete the individual temporary parquets
    # Check whether this puts the files in the bin or completely removes them
    for parquet_file in all_parquets:
        parquet_file.unlink()

    print('Creation of dataset completed. The temporary parquet files have been deleted.')
    logging.info(f'Creation of dataset completed. Time taken: {time.time() - start}. \
    The temporary parquet files have been deleted.')


def compare_formats(cases_df, data_dir, archive_name):
    """ Compares four formats on writing speed and file size. We test CSV, Feather, Parquet with snappy compression,
    and Parquet with brotli compression. The experiment only should be run for one year. The experimental files are
    saved with prefix exp_ to prevent the files from being included in create_final_dataset()
    """
    print('Starting comparison of file formats...')
    logging.info('Starting comparison of file formats...')

    # Save df as CSV
    start = time.time()
    cases_df.to_csv(
        data_dir / f'temp_cases_content{archive_name}.csv',
        mode='a',
        index=False,
        header=False)
    logging.info(f'Writing cases to csv took {time.time() - start}')

    # Save df as Feather
    start = time.time()
    cases_df.to_feather(data_dir / f'cases_content{archive_name}.feather')
    logging.info(f'Writing cases to feather took {time.time() - start}')

    # Save df as Parquet with snappy compression
    start = time.time()
    cases_df.to_parquet(data_dir / f'cases_content_{archive_name}_snappy.parquet')
    logging.info(f'Writing cases to parquet using snappy took {time.time() - start}')

    # Save df as Parquet with brotli compression
    start = time.time()
    cases_df.to_parquet(data_dir / f'cases_content{archive_name}_brotli.parquet', compression='brotli')
    logging.info(f'Writing cases to parquet using brotli took {time.time() - start}')

    print('Finished comparison of file formats.')
    logging.info('Finished comparison of file formats.')


if __name__ == '__main__':
    # Configure logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt = '%d-%m-%Y %H:%M:%S'
    script_name = Path(__file__).stem
    logging.basicConfig(
        filename=LOG_DIR / f'log_{script_name}.log',
        format=log_fmt,
        datefmt=date_fmt,
        level=logging.INFO
    )

    main()
