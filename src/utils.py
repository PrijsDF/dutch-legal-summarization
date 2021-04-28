import sys
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
REPORTS_DIR = ROOT_DIR / 'reports'
LOG_DIR = ROOT_DIR / 'reports/logs'


# ab = [1943, 1911, 1919, 2012, 2000, 2021, 3210]
# c = list(range(1910, 2022, 10))
#
# print(c)
#
# for a in ab:
#     print(int(a/10)*10)