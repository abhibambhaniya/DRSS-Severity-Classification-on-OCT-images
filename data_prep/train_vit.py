import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


import torchvision.models 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

import argparse
import os
import copy
import dataloader

def train_vit(args, device,batched_trainset, batched_testset ,num_class):
    # Define model
    model = torchvision.models.maxvit_t()
#     print(model)
    
    
    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    print(model.classifier)
    model.classifier = nn.Sequential( nn.AdaptiveAvgPool2d(output_size=1),
                                      nn.Flatten(start_dim=1, end_dim=-1),
                                      nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
                                      nn.Linear(in_features=512, out_features=512, bias=True),
                                      nn.Tanh(),
                                      nn.Linear(in_features=512, out_features=num_class, bias=False))
    model = model.to(device)

    print(model)
    # Tensorboard Writer
    writer = SummaryWriter('path_to_tensorboard_logs')

    # Train model
    num_epochs = args.epoch
    start = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0
        start_time = time.time()

        for images, labels in tqdm(batched_trainset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Evaluate model on test data
        test_loss = 0.0
        test_correct_predictions = 0.0
        test_total_predictions = 0.0
        with torch.no_grad():
            for images, labels in batched_testset:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Record test loss and accuracy
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total_predictions += labels.size(0)
                test_correct_predictions += (predicted == labels).sum().item()

        # Print statistics and add to Tensorboard
        end_time = time.time()
        epoch_time = end_time - start_time
        train_loss = running_loss / len(batched_trainset)
        train_accuracy = correct_predictions / total_predictions
        test_loss /= len(batched_testset)
        test_accuracy = test_correct_predictions / test_total_predictions

        print("Epoch: ", epoch + 1, "/", args.epoch, " Test Loss: ", test_loss, " Accuracy: ", test_accuracy, 
                " time spent: ", time.time() - start, " s")

    torch.save(model.state_dict(), args.save_pth)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '/usr/scratch/abhimanyu/courses/ECE8803_FML/OLIVES')
    parser.add_argument('--lr', type = int, default = 3e-3)
    parser.add_argument('--momentum', type = int, default = 0.9)
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--save_pth', type = str, default = '/usr/scratch/yangyu/FML_Model/vit')
  

    return parser.parse_known_args()


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args, unkown = parse_args()
    # Define transform
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_name = "vit_" + timestr + ".pth" 
    name = os.path.join(args.save_pth, base_name)
    args.save_pth = os.path.abspath(name)

    # # Load dataset
    # train_dataset = datasets.ImageFolder(root='path_to_train_folder', transform=transform)
    # test_dataset = datasets.ImageFolder(root='path_to_test_folder', transform=transform)

    # # Define dataloader
    batched_trainset, batched_testset = dataloader.dataloader(args, 'ResNet') 

    train_vit(args, device, batched_trainset, batched_testset, 3)


