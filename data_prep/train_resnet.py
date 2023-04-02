# Refer to https://github.com/kenshohara/3D-ResNets-PyTorch 
# Get pretrained model from: https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
from torchvision import models, transforms

import dataloader
import resnet as resnet

import pandas as pd
from PIL import Image
import argparse
import os
import copy
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args, batched_trainset, batched_testset, num_class):

    logfile = open(args.log, "w")

    logfile.write(str(args))
    logfile.write('\n')

    #rnet18 = models.resnet18(weights='DEFAULT')
    rnet18 = resnet.resnet_18() # untrained model taken from open-source github repo
    #print(rnet18)
    

    for param in rnet18.parameters():
        param.requires_grad = False

    # Replace the fully connected layers with new layers that include L2 regularization
    rnet18.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_class),
        #nn.Softmax(dim=1)
    )

    # k = rnet18.fc.in_features
    # rnet18.fc = nn.Linear(k, num_class)
    print(rnet18)
    rnet18 = rnet18.to(device)
    
    loss_fn = nn.CrossEntropyLoss() # loss function
    
    if (args.opt == 'AdamW'):
        optimizer = optim.AdamW(rnet18.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif (args.opt == 'SGD'):
        optimizer = optim.SGD(rnet18.parameters(), lr=args.lr, momentum=args.momentum)
    

    start = time.time()
    for epoch in range(args.epoch):
        # print("Running Epoch: ", epoch+1, "/", args.epoch,  " with learning rate: ", args.lr, " and momentum: ", args.momentum)
        
        rnet18.train() # train phase

        train_correct_num = 0
        train_total_num = 0
        train_loss = 0

        for train_data, train_label in tqdm(batched_trainset, desc=f"Epoch {epoch+1}/{args.epoch}"):
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            out = rnet18(train_data)

            loss = loss_fn(out, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, prediction = torch.max(out, dim=1)
            train_loss += loss.item()
            train_correct_num += (prediction == train_label).sum().item()
            train_total_num += train_label.size(0)

        train_loss = train_loss / len(batched_trainset)
        train_accuracy = train_correct_num / train_total_num

        train_msg = f'Epoch: {epoch + 1}/{args.epoch} Train Loss: {train_loss}, Accuracy: {train_accuracy}, time spent: {time.time() - start} s \n'
        print(train_msg)
        logfile.write(train_msg)

        rnet18.eval() # evaluation phase
        with torch.no_grad():
            test_loss = 0
            test_correct_num = 0
            test_total_num = 0

            for test_data, test_label in batched_testset:
                test_data = test_data.to(device)
                test_label = test_label.to(device)

                out = rnet18(test_data)

                _, prediction = torch.max(out, 1)
                loss = loss_fn(out, test_label)
                test_loss += loss.item()
            
                test_correct_num += (prediction == test_label).sum().item()
                test_total_num += test_label.size(0)

            test_loss = test_loss / len(batched_testset)
            test_accuracy = test_correct_num / test_total_num

            test_msg = f'Epoch: {epoch + 1}/{args.epoch} Test Loss: {test_loss}, Accuracy: {test_accuracy}, time spent: {time.time() - start} s \n'
            print(test_msg)
            logfile.write(test_msg)

    torch.save(rnet18.state_dict(), args.save_pth)
    logfile.close()




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train_features.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test_features.csv')
    parser.add_argument('--data_root', type = str, default = '')
    parser.add_argument('--opt', type = str, default = 'AdamW')
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type = float, default = 0.05)
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--epoch', type = int, default = 20)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--do_batch', type = bool, default = 1)
    parser.add_argument('--data_aug', type =int, default = 1)
    parser.add_argument('--log', type = str, default = '/usr/scratch/yangyu/FML_Model/resnet')
    parser.add_argument('--save_pth', type = str, default = '/usr/scratch/yangyu/FML_Model/resnet')

    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    base_name = "restnet18_" + timestr + ".pth"
    name = os.path.join(args.save_pth, base_name)
    args.save_pth = os.path.abspath(name)

    base_name_log = "restnet18_" + timestr + ".log"
    name_log = os.path.join(args.log, base_name_log)
    args.log = os.path.abspath(name_log)
    
    batched_trainset, batched_testset = dataloader.dataloader(args, 'ResNet')
    
    train(args, batched_trainset, batched_testset, 3)
