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

# transform_resnet = transforms.Compose([
#     transforms.Resize(size=(224,224)),
#     # transforms.ColorJitter(contrast=(0.5, 0.5)),
#     transforms.ToTensor(),
#     normalize,
# ])

transform_augment1 = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomRotation(degrees=(3, 3)),
    #transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.ColorJitter(contrast=(0.5, 0.5)),
    transforms.ToTensor(),
    normalize,
])

transform_augment2 = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomRotation(degrees=(3, 3)),
    transforms.GaussianBlur(kernel_size=(5, 5)),
    # transforms.ColorJitter(contrast=(0.5, 0.5)),
    transforms.ToTensor(),
    normalize,
])

transform_augment3 = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.GaussianBlur(kernel_size=(5, 5)),
    # transforms.ColorJitter(contrast=(0.5, 0.5)),
    transforms.ToTensor(),
    normalize,
])

transform_augment4 = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(p=1),
    # transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.ColorJitter(contrast=(0.5, 0.5)),
    transforms.ToTensor(),
    normalize,
])

transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    # transforms.ColorJitter(contrast=(0.5, 0.5)),
    transforms.ToTensor(),
    normalize,
])

  
transform_three_imgs = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # to compatible with resnet
    ])
class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,model = None):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        # self.annot['Severity_Label']
        temp = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        #self.annot['Severity_Label'] = temp + temp
        if (subset == 'train' and args.data_aug == 1):
            self.annot_labels = 5 * temp # 0: original. 1 - 5: rotate and color jitter. 6 - 10: rotate and gaussian
        else:
            self.annot_labels = temp
        
        self.model = model
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.transform_aug1 = transform_augment1
        self.transform_aug2 = transform_augment2
        self.transform_aug3 = transform_augment3
        self.transform_aug4 = transform_augment4
        self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        # self.path_list = self.annot['File_Path'].values
        self.path_list = self.annot['Volume_ID'].values
        self._labels = self.annot_labels
        # print(self._labels)
        self.label_freq = [(list(self._labels)).count(0), (list(self._labels)).count(1), (list(self._labels)).count(2)]
        # print(self.label_freq)
        #self._metadata 
        temp_meta = self.annot[['Leakage_Index', 'Age']].values.astype(np.float32)
        # print(temp_meta)
        self.meta_avg = np.mean(temp_meta, axis=0)
        # print(self.meta_avg)
        # if (subset == 'train'):
        #     self._metadata = temp_meta + temp_meta
        # else:
        self._metadata = temp_meta
        #assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        # img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]
        img_volume = []

        data_aug = 0
        if (self.subset == 'train' and index >= len(self.path_list)):
            data_aug = int(index / len(self.path_list))
            index = index % len(self.path_list)

        target = self._labels[index]
        metadata = self._metadata[index]
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
                if (self.subset == 'train'): 
                    img_volume.append(img_volume[i - 1])
                    continue
                else:
                    print('ERROR: Test Data missing frames')

            # img = transforms.functional.crop(img, top=80, height = 330 , width = 504, left = 0)

            if not self.model == "vit": 
                img = transform_three_imgs(img)

            if self.transform is not None and data_aug == 0:
                img = self.transform(img)
            elif self.transform_aug1 is not None and data_aug == 1 and self.subset == 'train':
                img = self.transform_aug1(img)
            elif self.transform_aug2 is not None and data_aug == 2 and self.subset == 'train':
                img = self.transform_aug2(img)
            elif self.transform_aug2 is not None and data_aug == 3 and self.subset == 'train':
                img = self.transform_aug3(img)
            elif self.transform_aug2 is not None and data_aug == 4 and self.subset == 'train':
                img = self.transform_aug4(img)

            img_volume.append(img)
        #img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        img_volume = torch.stack(img_volume, dim=1)
        return img_volume, target, metadata

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

    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    
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
                       cells_per_block=cells_per_block, visualize=False, transform_sqrt=True)
            frames.append(hog_features)
            # img = svm_transform(img)
            # img = np.array(img)
            # img_volume[index][i] = img.flatten()

        features = np.array(frames)
        normalized_features = scaler.fit_transform(features)
        pca = PCA(n_components=args.n_components)
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
    # if (model_name == 'ResNet'):
    #     trainset = OCTDataset(args, 'train', transform=transform_resnet, model=model_name)
    #     testset = OCTDataset(args, 'test', transform=transform_resnet, model=model_name)
    # else:
    trainset = OCTDataset(args, 'train', transform=transform, model=model_name)
    testset = OCTDataset(args, 'test', transform=transform, model=model_name)

    print(len(trainset))
    print(len(testset))
    print(args.do_batch)
    if (args.do_batch == 1):
        batched_trainset = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        batched_testset = DataLoader(testset, batch_size=args.batch_size, shuffle=True)
    else:
        batched_trainset = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        batched_testset = DataLoader(testset, batch_size=len(testset), shuffle=True)

    return batched_trainset, batched_testset, trainset.label_freq, testset.label_freq , trainset.meta_avg

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
