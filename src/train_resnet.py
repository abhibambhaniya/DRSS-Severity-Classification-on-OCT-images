# This is the script that train and test 3D ResNet18 base on OCT dataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score


import torchvision
from torchvision import models, transforms

import dataloader
import resnet18_3D as resnet

import pandas as pd
from PIL import Image
import argparse
import os
import copy
import time
from tqdm import tqdm
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model that combines ResNet and MLP
# Depending on the args.model_name, different network will be constructed
class ImageMetadataModel(nn.Module):
    def __init__(self, model_name='resnet18', num_class=3, dropout=0.5, num_meta=2):
        super(ImageMetadataModel, self).__init__()

        self.model_name = model_name
        if (model_name == 'resnet18'):
            self.resnet = resnet.resnet_18()

            self.resnet.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(256, num_class),
            )

        elif (model_name == 'resnet18+meta'):
            self.resnet = resnet.resnet_18(n_outputs=32)
            num_features = 32

            # Define MLP for metadata
            self.metadata_mlp = nn.Sequential(
                nn.Linear(num_meta, 4),
                nn.ReLU(inplace=True),
                nn.Linear(4, 4),
                nn.ReLU(inplace=True),
            )

            # Define final MLP layers
            self.final_mlp = nn.Sequential(
                nn.Linear(num_features + 4, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(16, 3),
                nn.Softmax(dim=1)
            )
        
    def forward(self, image_data, metadata):
        if (self.model_name == 'resnet18'):
            output = self.resnet(image_data)
        elif (self.model_name == 'resnet18+meta'):
            image_features = self.resnet(image_data)
            metadata_features = self.metadata_mlp(metadata)
            combined_features = torch.cat([image_features, metadata_features], dim=1)
            output = self.final_mlp(combined_features)
        
        return output


# train 3D ResNet18, and writing the best test balanced accuracy result in pt file, and save the running log
def train(args, batched_trainset, batched_testset, weight, train_meta_avg, num_class):

    logfile = open(args.log, "w")

    logfile.write(str(args))
    logfile.write('\n')

    if (args.meta == 1):
        print('Model: Resnet18 + Meta')
        logfile.write('Model: Resnet18 + Meta\n')
        model = ImageMetadataModel(model_name='resnet18+meta', num_class=num_class, dropout = args.dropout, num_meta=args.num_meta)
    else:
        print('Model: Resnet18')
        logfile.write('Model: Resnet18\n')
        model = ImageMetadataModel(model_name='resnet18', num_class=num_class, dropout = args.dropout)
    
    model = model.to(device)
    
    if (args.weighted_loss == 1):
        weight = weight.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight) # loss function
        print('doing weighted loss')
        logfile.write('doing weighted loss\n')
    else:
        loss_fn = nn.CrossEntropyLoss() # loss function
        print('doing unweighted loss')
        logfile.write('doing unweighted loss\n')

    if (args.opt == 'AdamW'):
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif (args.opt == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    logfile.close()

    best_balanced_accuracy = 0.0
    best_epoch_msg = None
    best_model = None
    best_pred = {}
    start = time.time()
    for epoch in range(args.epoch):
        logfile = open(args.log, "a")
        
        model.train() # train phase

        train_correct_num = 0
        train_total_num = 0
        train_loss = 0
        train_balanced_predict = []
        train_balanced_true = []

        for train_data, train_label, metadata in tqdm(batched_trainset, desc=f"Epoch {epoch+1}/{args.epoch}"):
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            # check if NaN number is showing up for indexed metadata
            has_nan = torch.isnan(metadata)
            any_nan = torch.any(has_nan)
            if (any_nan):
                metadata = torch.zeros(metadata.shape, device= device)
            else:
                metadata = metadata.to(device)

            # take the output and do backward propogation
            out = model(train_data, metadata)
            loss = loss_fn(out, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # take the prediction, and accumulate data for later collection
            _, prediction = torch.max(out, dim=1)
            train_loss += loss.item()
            train_correct_num += (prediction == train_label).sum().item()
            train_total_num += train_label.size(0)
            
            train_balanced_predict.extend(prediction.detach().cpu().numpy().tolist())
            train_balanced_true.extend(train_label.detach().cpu().numpy().tolist())

        train_loss = train_loss / len(batched_trainset)
        train_accuracy = train_correct_num / train_total_num
        train_balanced_accuracy = balanced_accuracy_score(train_balanced_true, train_balanced_predict)

        train_msg = f'Epoch: {epoch + 1}/{args.epoch} Train Loss: {train_loss}, Accuracy: {train_accuracy}, Balanced Accuracy: {train_balanced_accuracy}, time spent: {time.time() - start} s \n'
        print(train_msg)
        logfile.write(train_msg)

        model.eval() # evaluation phase
        with torch.no_grad():
            test_loss = 0
            test_correct_num = 0
            test_total_num = 0
            test_balanced_predict = []
            test_balanced_true = []

            for test_data, test_label, _ in batched_testset:
                test_data = test_data.to(device)
                test_label = test_label.to(device)
                metadata = torch.zeros(_.shape, device= device)

                out = model(test_data, metadata)

                _, prediction = torch.max(out, 1)
                loss = loss_fn(out, test_label)
                test_loss += loss.item()
                
                test_correct_num += (prediction == test_label).sum().item()
                test_total_num += test_label.size(0)

                test_balanced_predict.extend(prediction.detach().cpu().numpy().tolist())
                test_balanced_true.extend(test_label.detach().cpu().numpy().tolist())

            test_loss = test_loss / len(batched_testset)
            test_accuracy = test_correct_num / test_total_num
            test_balanced_accuracy = balanced_accuracy_score(test_balanced_true, test_balanced_predict)

            test_msg = f'Epoch: {epoch + 1}/{args.epoch} Test Loss: {test_loss}, Accuracy: {test_accuracy}, Balanced Accuracy: {test_balanced_accuracy}, time spent: {time.time() - start} s \n'
            print(test_msg)
            logfile.write(test_msg)
            
            if (best_balanced_accuracy < test_balanced_accuracy):
                best_balanced_accuracy = test_balanced_accuracy
                best_epoch_msg = test_msg
                best_model = copy.deepcopy(model.state_dict())
                best_pred['label'] = test_balanced_true
                best_pred['prediction'] = test_balanced_predict

        # save the best model every three epochs
        if epoch % 3 == 0:
            logfile.write('Checkpoint: Saving the model with the best test balanced accuracy....')
            logfile.write(best_epoch_msg)
            torch.save(best_model, args.save_pt)
            with open(args.save_pred,'wb') as f:
                pickle.dump(best_pred, f)

        logfile.close()

    logfile = open(args.log, "a")
    logfile.write('Last Checkpoint: Saving the model with the best test balanced accuracy....')
    logfile.write(best_epoch_msg)
    logfile.close()
    
    # after all epochs, save the best model
    torch.save(best_model, args.save_pt)
    
    with open(args.save_pred,'wb') as f:
        pickle.dump(best_pred, f)
    

# test-only function, taken the pt file, the function will run a prediction on the test dataset
def test_model(model_val, batched_testset, num_class):
    
    if (args.meta == 1):
        print('Model: Resnet18 + Meta')
        model = ImageMetadataModel(model_name='resnet18+meta', num_class=num_class, dropout = args.dropout)
    else:
        print('Model: Resnet18')
        model = ImageMetadataModel(model_name='resnet18', num_class=num_class, dropout = args.dropout)
    
    model.load_state_dict(model_val)

    model = model.to(torch.device('cuda:0'))
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_correct_num = 0
        test_total_num = 0
        test_balanced_predict = []
        test_balanced_true = []

        for test_data, test_label, _ in batched_testset:
            test_data = test_data.to(torch.device('cuda:0'))
            test_label = test_label.to(torch.device('cuda:0'))
            metadata = torch.zeros(_.shape, device= torch.device('cuda:0'))

            out = model(test_data, metadata)

            _, prediction = torch.max(out, 1)
            
            test_correct_num += (prediction == test_label).sum().item()
            test_total_num += test_label.size(0)

            test_balanced_predict.extend(prediction.detach().cpu().numpy().tolist())
            test_balanced_true.extend(test_label.detach().cpu().numpy().tolist())

        test_accuracy = test_correct_num / test_total_num
        test_balanced_accuracy = balanced_accuracy_score(test_balanced_true, test_balanced_predict)

        test_msg = f'Test Loss: {test_loss}, Accuracy: {test_accuracy}, Balanced Accuracy: {test_balanced_accuracy} \n'
        print(test_msg)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train_features.csv', help="The volume granularity trainset csv")
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test_features.csv', help="The volume granularity testset csv")
    parser.add_argument('--data_root', type = str, default = '', help="The root of where data locate")
    parser.add_argument('--model_test', type =int, default = 0, help="Directly load model and run on test data")
    parser.add_argument('--model_name', type = str, default = '', help="The model to load for testing")
    parser.add_argument('--seed', type = int, default = 8803, help='The manual seed to specify')
    parser.add_argument('--opt', type = str, default = 'AdamW', help='The optimizer, AdamW/SGD are available')
    parser.add_argument('--lr', type = float, default = 0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0.05, help='Weight decay for AdamW')
    parser.add_argument('--momentum', type = float, default = 0.9, help='Momentum for SGD')
    parser.add_argument('--dropout', type =float, default = 0.5, help='Dropout value for the model fc layer')
    parser.add_argument('--epoch', type = int, default = 20, help='The number of epochs')
    parser.add_argument('--batch_size', type = int, default = 8, help='Batch Size')
    parser.add_argument('--do_batch', type = int, default = 1, help='Do batch or not')
    parser.add_argument('--meta', type = int, default = 1, help='Use metadata or not')
    parser.add_argument('--num_meta', type = int, default = 2, help='The number of meta feauters, look at dataloader to see more. Values: 2/9')
    parser.add_argument('--weighted_loss', type = int, default = 1, help='Do weighted loss or not')
    parser.add_argument('--data_aug', type =int, default = 1, help='Add data augmentation or not')
    parser.add_argument('--log', type = str, default = '/usr/scratch/yangyu/FML_Model/resnet', help='Where the log of running store')
    parser.add_argument('--save_pt', type = str, default = '/usr/scratch/yangyu/FML_Model/resnet', help='Where the pt file store')
    parser.add_argument('--save_pred', type = str, default = '/usr/scratch/yangyu/FML_Model/resnet', help='Where the prediction result store')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    # use time string so that every run will have unique file name
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    base_name = "restnet18_" + timestr + ".pt"
    name = os.path.join(args.save_pt, base_name)
    args.save_pt = os.path.abspath(name)

    label_base_name = "restnet18_predictlabel_" + timestr + ".pickle"
    save_label = os.path.abspath(os.path.join(args.save_pred, label_base_name))
    args.save_pred = os.path.abspath(save_label)

    base_name_log = "restnet18_" + timestr + ".log"
    name_log = os.path.join(args.log, base_name_log)
    args.log = os.path.abspath(name_log)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    batched_trainset, batched_testset, train_freq, test_freq, train_meta_avg = dataloader.dataloader(args, 'ResNet')

    freq = np.array(train_freq)
    # weight = freq / np.sum(freq)
    weight = [sum(freq) / (3 * count) for count in freq]

    weight = torch.tensor(weight, dtype=torch.float)
    print(weight)

    if (args.model_test == 1):
        model_val = torch.load(args.model_name, map_location=torch.device('cuda:0'))
        test_model(model_val, batched_testset, 3)
    else:
        train(args, batched_trainset, batched_testset, weight, train_meta_avg, 3)
