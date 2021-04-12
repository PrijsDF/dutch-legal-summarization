from bs4 import BeautifulSoup
from datetime import datetime
from zipfile import ZipFile
import glob
import pandas as pd
from tqdm import trange
import os
import logging

from rechtspraak_helper_functions import parse_xml, get_document_attributes

# pd.set_option('display.max_columns', 500)
# pd.set_option('max_colwidth', 400)
# pd.options.display.width=None


def main():
    """ Runs data processing scripts to turn external data from (../external) into
        a raw dataset (saved in ../raw) that can be used as a starting point for creating the
        final dataset.
    """
    # Start logging
    logger = logging.getLogger(__name__)
    logger.info('Making raw dataset from external Rechtspraak data')

    # Specify years; 1912 does not exist
    years = list(range(1911, 2022))
    years.remove(1912)

    years = range(2020, 2021)

    # collection_zip = f'{DATA_DIR}/external/OpenDataUitspraken.zip'
    # collection_archive = ZipFile(collection_zip, 'r')
    # collection_filenames = sorted(collection_archive.namelist())
    #
    # yearie = '2018'
    # current = [year for year in collection_filenames if year.startswith(yearie)]
    #
    # archive = ZipFile(current[0], 'r')
    # archive_filenames = archive.namelist()

    for i in trange(len(years), desc='Total years'):
        current_year = years[i]
        #print(glob.glob(f'{DATA_DIR}/external/OpenDataUitspraken/{str(current_year)}/*'))
        # Get all zip file names
        zip_files = glob.glob(f'{DATA_DIR}/external/OpenDataUitspraken/{str(current_year)}/*')

        for j in trange(len(zip_files), desc='Archives in ' + str(current_year), leave=False):
            # Store all cases of current archive in these dfs
            cases_meta_df = pd.DataFrame(columns=["identifier",
                                                  "missing_parts",
                                                  "case_type",
                                                  "case_number",
                                                  "jurisdiction",
                                                  "creator",
                                                  "judgment_date",
                                                  "relation",
                                                  "procedures",
                                                  "seat_location",
                                                  "references",
                                                  "publisher",
                                                  "issue_date",
                                                  "modified"
                                                  # "language",
                                                  # "format"
                                                  ])

            cases_content_df = pd.DataFrame(columns=["identifier",
                                                     "summary",
                                                     "description"])
            zip_file = zip_files[j]

            # Make archive of the zip file
            archive = ZipFile(zip_file, 'r')
            archive_filenames = archive.namelist()

            for k in trange(len(archive_filenames), desc='Current archive', leave=False):
                file_name = archive_filenames[k]

                # Read the content of the zip file (XML) into bf4 parser
                case_rdf, case_summary, case_description = parse_xml(archive.read(file_name))

                # Parse case_rdf to get document attributes
                case_meta_info = get_document_attributes(case_rdf)

                # Will store the case description, summary and identifier
                case_content = {}

                # We will use the identifier, or ecli, as the primary key between both dfs
                case_content["identifier"] = case_meta_info["identifier"]

                # Will be stored as meta information
                missing = tuple()

                # Find the summary of the document
                if case_summary is not None:
                    case_content["summary"] = case_summary.get_text("|||", strip=True)
                else:
                    case_content["summary"] = "NOT_FOUND"
                    missing = missing + ("summary",)

                # Find the full description of the case
                if case_description is not None:
                    case_content["description"] = case_description.get_text("|||", strip=True)
                else:
                    case_content["description"] = "NOT_FOUND"
                    missing = missing + ("description",)

                # Add missing parts information to meta df
                case_meta_info["missing_parts"] = missing

                # Append to df
                cases_meta_df = cases_meta_df.append(case_meta_info, ignore_index=True)
                cases_content_df = cases_content_df.append(case_content, ignore_index=True)

            # Finally, save the fresh dfs to csv files
            year_path = f'{DATA_DIR}/raw/CSV per year/{str(current_year)}'

            # Cases meta
            cases_meta_df.to_csv(
                year_path + "_cases_meta.csv",
                mode='a',
                index=False,
                header=not os.path.exists(year_path + "_cases_meta.csv"))

            # Cases content
            cases_content_df.to_csv(
                year_path + "_cases_content.csv",
                mode='a',
                index=False,
                header=not os.path.exists(year_path + "_cases_content.csv"))

            logging.info(f'{current_year} - {i + 1} has been parsed and saved')


if __name__ == '__main__':
    ROOT_DIR = '../..'
    DATA_DIR = f'{ROOT_DIR}/data'
    LOG_DIR = f'{ROOT_DIR}/reports/logs'

    # Configurate logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(
        filename=f'{LOG_DIR}/log{__name__}.log',
        format=log_fmt,
        datefmt=date_fmt,
        level=logging.INFO
    )

    main()

# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
#
#
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')
#
#
# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
#
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
#
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
#
#     main()
