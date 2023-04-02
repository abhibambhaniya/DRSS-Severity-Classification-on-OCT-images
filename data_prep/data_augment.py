import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy


def data_aug():






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '')
    parser.add_argument('--data_aug_root', type = str, default = '')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    data_aug()