import sys
from pathlib import Path

ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'reports/logs'

# before this method can work, the logger needs to be included as a param
# def print_and_log(text):
#     print(text)
#     logging.info(text)
