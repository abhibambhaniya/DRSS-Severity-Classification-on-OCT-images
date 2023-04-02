import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
visualize = False
transform_sqrt = True

n_components = 20



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

transform_resnet = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.Grayscale(num_output_channels=3), # to compatible with resnet
    transforms.ToTensor(),
    normalize,
])

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
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
        # self.path_list = self.annot['File_Path'].values
        self.path_list = self.annot['Volume_ID'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        # img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]
        img_volume = []

        target = self._labels[index]

        folder_path = self.root + self.path_list[index]
        
        # there are maximum 49 frames per volume ID, concatenate them here for 3D CNN
        # if certain frame did not exsit, simply take a copy of previous frame
        for i in range(0, 49): 
            tif = str(i) + '.tif'
            png = str(i) + '.png'
            
            if (os.path.isfile(os.path.join(folder_path, tif))):
               img = Image.open(os.path.join(folder_path, tif)).convert("L")
            elif (os.path.isfile(os.path.join(folder_path, png))):
               img = Image.open(os.path.join(folder_path, png)).convert("L")
            else:
                img_volume.append(img_volume[i - 1])
                continue

            if self.transform is not None:
                img = self.transform(img)

            img_volume.append(img)

        #img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        img_volume = torch.stack(img_volume, dim=1)
        return img_volume, target

    def __len__(self):
        return len(self._labels)     


def image_feature_extraction(args, data_type):
    if data_type == 'train':
        annot = pd.read_csv(args.annot_train_prime)
    elif data_type == 'test':
        annot = pd.read_csv(args.annot_test_prime)

    annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(annot['DRSS'].values)]
    root = os.path.expanduser(args.data_root)
    nb_classes=len(np.unique(list(LABELS_Severity.values())))
    path_list = annot['Volume_ID'].values

    labels = annot['Severity_Label'].values
    print(labels)
    assert len(path_list) == len(labels)

    # get features
    # img_volume = np.zeros((len(labels), 49, 224*224), dtype=object)
    #img_volume.fill([])
    # features = np.zeros((len(labels), 49))
    img_volume = []
    
    scaler = StandardScaler()
    for index in range(len(path_list)):
        # img_volume[index] = []
        frames = []
        folder_path = root + path_list[index]
        
        # there are maximum 49 frames per volume ID, concatenate them here for 3D CNN
        # if certain frame did not exsit, simply take a copy of previous frame
        for i in range(0, 49): 
            tif = str(i) + '.tif'
            png = str(i) + '.png'
            
            if (os.path.isfile(os.path.join(folder_path, tif))):
                img = Image.open(os.path.join(folder_path, tif)).convert("L")
            elif (os.path.isfile(os.path.join(folder_path, png))):
                img = Image.open(os.path.join(folder_path, png)).convert("L")
            else:
                img = frames[i - 1]
                frames.append(frames[i - 1])
                continue
    
            hog_features = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, visualize=visualize, transform_sqrt=transform_sqrt)
            frames.append(hog_features)
            # img = svm_transform(img)
            # img = np.array(img)
            # img_volume[index][i] = img.flatten()

        features = np.array(frames)
        normalized_features = scaler.fit_transform(features)
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(normalized_features)

        img_volume.append(reduced_features)
    
    print(np.shape(img_volume))
    img_volume = np.array(img_volume)
    ret = img_volume.reshape((len(img_volume), -1))
    print(np.shape(ret))


    # img_volume = np.array(img_volume)
    # print(len(labels))
    # print(np.shape(img_volume))

    return ret, labels


def svm_dataloader(args, model_name):
    # this can also be used for another classification we choose
    if (model_name != 'SVM'):
        print("The provided model is not SVM")
        return

    train_features, train_labels = image_feature_extraction(args, 'train')
    test_features, test_labels = image_feature_extraction(args, 'test')

    return train_features, train_labels, test_features, test_labels


def dataloader(args, model_name):
    if (model_name == 'ResNet'):
        trainset = OCTDataset(args, 'train', transform=transform_resnet)
        testset = OCTDataset(args, 'test', transform=transform_resnet)
    else:
        trainset = OCTDataset(args, 'train', transform=transform)
        testset = OCTDataset(args, 'test', transform=transform)

    print(len(trainset))
    print(len(testset))
    print(args.do_batch)
    if (args.do_batch == 1):
        print('a')
        batched_trainset = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        batched_testset = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        batched_trainset = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        batched_testset = DataLoader(testset, batch_size=len(testset), shuffle=True)

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
