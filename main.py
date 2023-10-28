import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
import warnings
from manage_data import *

warnings.filterwarnings('ignore')
from IPython.display import display

import matplotlib.pyplot as plt
# %matplotlib inline



if __name__ == '__main__':
    path = 'C:/Users/user/Documents/Python/HCM_finder/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate = 100
    #load_and_convert_data(path, sampling_rate)
    #load_data_for_diagnostics(path)
    reformat_data_for_EDA(path, sampling_rate)
