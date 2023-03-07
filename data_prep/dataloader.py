import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy


LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.Grayscale(num_output_channels=3), # to compatible with resnet
    transforms.ToTensor(),
    normalize,
])
    
class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        # img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._labels)     


def dataloader(args):
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)

    if (args.batch_size != 1):
        batched_trainset = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        batched_testset = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        batched_trainset = trainset
        batched_testset = testset

    return batched_trainset, batched_testset

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
#     parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
#     parser.add_argument('--data_root', type = str, default = '')
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_args()
#     trainset = OCTDataset(args, 'train', transform=transform)
#     testset = OCTDataset(args, 'test', transform=transform)
#     print(trainset[1][0].shape)
#     print(len(trainset), len(testset))
