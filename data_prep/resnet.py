import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
from torchvision import models, transforms

import dataloader

import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(args, batched_trainset, batched_testset, num_class):
    rnet18 = models.resnet18(weights='DEFAULT')
    
    loss_fn = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(rnet18.parameters(), lr=args.lr, momentum=args.momentum)
    
    k = rnet18.fc.in_features
    rnet18.fc = nn.Linear(k, num_class)
    rnet18 = rnet18.to(device)

    start = time.time()
    for epoch in range(args.epoch):
        print("Running Epoch: ", epoch, "/", args.epoch,  " with learning rate: ", args.lr, " and momentum: ", args.momentum)
        
        rnet18.train() # train phase

        correct_num = 0
        total_num = 0
        epoch_loss = 0

        for batch_id, (train_data, label) in enumerate(batched_trainset):
            train_data = train_data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = rnet18(train_data)
            _, prediction = torch.max(out, dim=1)

            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * train_data.size(0)
            correct_num += torch.sum(prediction==label)

        epoch_loss = epoch_loss / len(batched_trainset)
        epoch_accuracy = correct_num / len(batched_trainset)
        print("Epoch: ", epoch, "/", args.epoch, " Training Loss: ", epoch_loss, " Accuracy: ", epoch_accuracy, 
               " time spent: ", time.time() - start, " s")


        rnet18.eval() # evaluation phase
        with torch.no_grad():
            test_loss = 0
            test_correct_num = 0

            for test_data, test_label in batched_testset:
                test_data = test_data.to(device)
                test_label = test_label.to(device)

                out = rnet18(test_data)

                _, prediction = torch.max(out, 1)
                loss = loss_fn(out, test_label)
                test_loss += loss * test_data.size(0)
            
                test_correct_num += torch.sum(prediction==test_label)

            test_loss = test_loss / len(batched_testset)
            test_accuracy = test_correct_num / len(batched_testset)

        print("Epoch: ", epoch, "/", args.epoch, " Test Loss: ", test_loss, " Accuracy: ", test_accuracy, 
                " time spent: ", time.time() - start, " s")

    torch.save(rnet18.state_dict(), args.save_pth)





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '')
    parser.add_argument('--lr', type = int, default = 0.001)
    parser.add_argument('--momentum', type = int, default = 0.9)
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--save_pth', type = str, default = '../models/')



    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_name = "restnet18_" + timestr + ".pth" 
    name = os.path.join(args.save_pth, base_name)
    args.save_pth = os.path.abspath(name)
    
    batched_trainset, batched_testset = dataloader.dataloader(args)    

    print(len(batched_trainset), len(batched_testset))

    train(args, batched_trainset, batched_testset, 3)
