import os
import logging
import time

from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.utils import ROOT_DIR, DATA_DIR, LOG_DIR

from rechtspraak_helper_functions import parse_xml, get_document_attributes


def main():
    """ Runs data processing scripts to turn external data from (../external) into
        a raw dataset (saved in ../raw) that can be used as a starting point for creating the
        final dataset.
    """
    # Start logging
    logging.info('Making raw dataset from external Rechtspraak data')

    # Specify years; 1912 does not exist
    years = list(range(1911, 2022))
    years.remove(1912)
    years = range(2020, 2021)  # Temp

    # Get month archives of each year; we will loop over these
    all_month_archives = []
    for year in years:
        year_path = DATA_DIR / f'external/OpenDataUitspraken/{str(year)}'
        month_archives = [x for x in year_path.iterdir() if x.is_file()]
        all_month_archives += month_archives

    # Start processing all archives
    pbar = tqdm(all_month_archives)
    for month_archive in pbar:
        # Config progress bar
        year = month_archive.stem[:4]
        month = month_archive.stem[4:]
        pbar.set_description(f'Processing month {month} of {year}')

        # Store all cases of current archive in these dfs
        cases_meta_df = pd.DataFrame(columns=[
            'identifier', 'missing_parts', 'case_type', 'case_number', 'jurisdiction',
            'creator', 'judgment_date', 'relation', 'procedures', 'seat_location',
            'references', 'publisher', 'issue_date', 'modified',  # 'language', # 'format'
        ])

        cases_content_df = pd.DataFrame(columns=['identifier', 'summary', 'description'])

        # Make archive containing all the legal cases of the month's zip file
        archive = ZipFile(month_archive, 'r')
        cases_archive = archive.namelist()

        for legal_case in cases_archive:
            # Read the content of the zip file (XML) into bf4 parser
            case_rdf, case_summary, case_description = parse_xml(archive.read(legal_case))

            # Parse case_rdf to get document attributes
            case_meta_info = get_document_attributes(case_rdf)

            # We store the case description, summary and identifier in case_content
            # We will use the identifier, or ecli, as the primary key between both dfs
            case_content = {'identifier': case_meta_info['identifier']}

            # Will be stored as meta information
            missing = tuple()

            # Find the summary of the document
            if case_summary is not None:
                case_content['summary'] = case_summary.get_text('|', strip=True)
            else:
                case_content['summary'] = 'none'
                missing = missing + ('summary',)

            # Find the full description of the case
            if case_description is not None:
                case_content['description'] = case_description.get_text('|', strip=True)
            else:
                case_content['description'] = 'none'
                missing = missing + ('description',)

            # Add missing parts information to meta df
            case_meta_info['missing_parts'] = missing

            # Append to df
            cases_meta_df = cases_meta_df.append(case_meta_info, ignore_index=True)
            cases_content_df = cases_content_df.append(case_content, ignore_index=True)

        # The first time we write to the csv, we want to include the header
        header = False
        if month_archive == all_month_archives[0]:
            header = True

        # Finally, save the fresh dfs to csv files
        save_path = DATA_DIR / f'raw/OpenDataUitspraken'

        # Temp; debug
        archive_namej = month_archive.stem if len(years) == 1 else ''

        archive_name = month_archive.stem

        # Cases meta
        cases_meta_df.to_csv(
            save_path / f'cases_meta{archive_name}.csv',
            mode='w' if header else 'a',
            index=False,
            header=header)

        # Temp
        # logging.info(f'{archive_namej} contains {len(cases_archive)} files.')

        # Cases content
        # start = time.time()
        # cases_content_df.to_feather(save_path / f'cases_content{archive_name}.feather')
        # logging.info(f'Writing cases to feather took {time.time() - start}')

        start = time.time()
        cases_content_df.to_parquet(save_path / f'cases_content{archive_name}_snappy.parquet')
        logging.info(f'Writing cases to parquet took {time.time() - start}')

        # start = time.time()
        # cases_content_df.to_parquet(save_path / f'cases_content{archive_name}_brotli.parquet', compression='brotli')
        # logging.info(f'Writing cases to parquet took {time.time() - start}')

        # start = time.time()
        # cases_content_df.to_csv(
        #     save_path / f'cases_content{archive_name}.csv',
        #     mode='w',  # 'w' if header else 'a',
        #     index=False,
        #     header=header)
        # logging.info(f'Writing cases to csv took {time.time() - start}')

        logging.info(f'{archive_name} has been parsed and saved')

# Needed later: (https://stackoverflow.com/a/52193992)
# from pathlib import Path
# import pandas as pd
#
# data_dir = Path('dir/to/parquet/files')
# full_df = pd.concat(
#     pd.read_parquet(parquet_file)
#     for parquet_file in data_dir.glob('*.parquet')
# )
# full_df.to_csv('csv_file.csv')


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
