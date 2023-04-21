import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import sklearn 

import torchvision.models 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from torchinfo import summary
import argparse
import os
import copy
import dataloader

import pickle

def Severity_to_DRRS( serverity):
#     DRSS = []
#     for i in serverity:
#         if i <= 1:
#             DRSS.append(0)
#         elif i<= 3:
#             DRSS.append(1)
#         else:
#             DRSS.append(2)
# 
#     return torch.FloatTensor(DRSS)
    return serverity
# Define the model architecture
class ImageMetadataClassifier(nn.Module):
    def __init__(self, num_class):
        super(ImageMetadataClassifier, self).__init__()
#         vit_model = torchvision.models.vit_b_16(,dropout=0.2)
        vit_model = torchvision.models.VisionTransformer(
                image_size = 224,
                patch_size=16,
                num_layers=3,
                num_heads=8,
                hidden_dim=512,
                mlp_dim=2048,
                dropout=0.0,
                num_classes = 32)
        vit_model.conv_proj = nn.Conv2d(49 , 512 , kernel_size=(16,16), stride=(16,16))
#         vit_model.encoder.layers = nn.Sequential( *list(vit_model.encoder.layers.children()))      ## Have only 2 Encoders
#         vit_model.heads.head = nn.Linear(in_features=256, out_features=32, bias=True)

        self.image_features = vit_model
#         mnasnet = torchvision.models.mnasnet1_0(num_classes = 32)
#         self.image_features = nn.Sequential( nn.Conv2d(49 , 3 , kernel_size=(1,1), stride=(1,1)),
#                 mnasnet
#         )
        self.metadata_fc = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                # Add more layers as needed
                )
        self.classifier = nn.Linear(32 + 4, num_class)

    def forward(self, x, metadata):
#         print(x.dtype,metadata.dtype)
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        metadata_extract = self.metadata_fc(metadata)
        x = torch.cat([x, metadata_extract], dim=1)
        x = self.classifier(x)
        return x


def train_dnn(args, device,batched_trainset, batched_testset, weight, train_meta_avg, num_class):
    # Define model
    model = ImageMetadataClassifier(num_class)
    print(model)

    if( args.load_checkpoint is not None):
        try:
            print("Trying to Load Checkpoint")
            model.load_state_dict(torch.load(args.load_checkpoint))
            print("Load Checkpoint Successful")
        except:
            print("Checkpoint load failed")

    # Define optimizer and loss function
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight) # loss function
#     criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW( model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
    ## Autoscaling while training can speedup train time on newer GPUs like V100 and A100
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ## Define LR schedular
    if(args.lr_scheduler):
        print("Enabling LR scheduler")
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                                optimizer, start_factor=0.01, total_iters=args.lr_warmup_epochs
                                            )

        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=args.epoch - args.lr_warmup_epochs, eta_min=args.lr_min
                                    )

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
                                    )


    ## Updating the classifier from imagenet 1k classification into num_class classification
    model = model.to(device)
    # Train model
    num_epochs = args.epoch
    start = time.time()

    best_test_accuracy = 0 
    best_test_balanced_accuracy = 0 
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0
        start_time = time.time()
        model.train()
        labels_all =  [ ]
        predicted_all = [ ]

#         # Train 1 epoch
#         for images, labels, metadata in tqdm(batched_trainset, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             # Move data and labels to device
#             images = images.to(device)
#             labels = labels.to(device)
#             metadata = metadata.to(device)
#             images = torch.flatten(images , start_dim = 1 , end_dim =2)

#             has_nan = torch.isnan(metadata)
#             any_nan = torch.any(has_nan)
#             if (any_nan):
#                 metadata = torch.zeros(metadata.shape, device= device)
#             else:
#                 metadata = metadata.to(device)

#             # Forward pass
#             with torch.cuda.amp.autocast(enabled=scaler is not None):
#                 outputs = model(x = images, metadata = metadata)
# #                 print(outputs)
#                 loss = criterion(outputs, labels)


#             # Backward pass
#             optimizer.zero_grad()
#             if scaler is not None:
#                 scaler.scale(loss).backward()
#                 scaler.unscale_(optimizer)
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)       ## 1.0 is args.clip_grad_norm
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(model.parameters(), 1.0)       ## 1.0 is args.clip_grad_norm
#                 optimizer.step()

#             # Record training loss and accuracy
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total_predictions += labels.size(0)
# #             print(predicted.type(),labels.type())
#             predicted = Severity_to_DRRS(predicted)
#             labels = Severity_to_DRRS(labels)
# #             print(predicted.type(),labels.type())
#             correct_predictions += (predicted == labels).sum().item()
#             labels_all.append(labels)
#             predicted_all.append(predicted)
# #             print(labels_all,predicted_all)

# #         print(np.shape(labels_all),np.shape(predicted_all))
#         labels_all =  torch.cat(labels_all, dim=0).cpu()
#         predicted_all =  torch.cat(predicted_all, dim=0).cpu()
#         unique_labels, counts_labels = np.unique(labels_all, return_counts=True)
#         unique_predicted, counts_predicted = np.unique(predicted_all, return_counts=True)

# #         print(labels_all,predicted_all)
#         print("Train Balanced accuracy: ", sklearn.metrics.balanced_accuracy_score(labels_all, predicted_all)) 
#         print(f'Train output distribution for labels {unique_labels} : {counts_labels} , predicted {unique_predicted} : {counts_predicted}') 
        
        
#         if(args.lr_scheduler):
#             lr_scheduler.step()

        if epoch % 1 == 0:          ## Test every epoch, can change according to requirements
            # Evaluate model on test data
            test_loss = 0.0
            test_correct_predictions = 0.0
            test_total_predictions = 0.0
    
            labels_all =  [ ]
            predicted_all = [ ]
            model.eval()
            with torch.no_grad():
                for images, labels, _ in batched_testset:
                    images = images.to(device)
                    labels = labels.to(device)
                    metadata = torch.float(train_meta_avg, device= device)

                    images = torch.flatten(images , start_dim = 1 , end_dim =2)
                    # Forward pass
                    outputs = model(images,metadata)
                    loss = criterion(outputs, labels)
    
                    # Record test loss and accuracy
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total_predictions += labels.size(0)
                    predicted = Severity_to_DRRS(predicted)
                    labels = Severity_to_DRRS(labels)
                    test_correct_predictions += (predicted == labels).sum().item()
                    labels_all.append(labels)
                    predicted_all.append(predicted)
#             print(labels_all,predicted_all)

                labels_all =  torch.cat(labels_all, dim=0).cpu()
                predicted_all =  torch.cat(predicted_all, dim=0).cpu()
                test_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(labels_all, predicted_all)
                unique_labels, counts_labels = np.unique(labels_all, return_counts=True)
                unique_predicted, counts_predicted = np.unique(predicted_all, return_counts=True)
                with open("ViT test best.pickle",'wb') as f:
                    pickle.dump(predicted_all, f)
#         print(labels_all,predicted_all)
                print(" Test output distribution for labels, predicted: ",counts_labels, counts_predicted) 
                print(f'Train output distribution for labels {unique_labels} : {counts_labels} , predicted {unique_predicted} : {counts_predicted}') 
                print(" Test Balanced accuracy: ", test_balanced_accuracy) 
 #                     print(" predicted :", predicted)
#                     print(" labels :", labels)
            test_loss /= len(batched_testset)
            test_accuracy = test_correct_predictions / test_total_predictions
            print(" Test Loss: ", test_loss, " Accuracy: ", test_accuracy)

        # Print statistics and add to Tensorboard
        end_time = time.time()
        if best_test_accuracy < test_accuracy:
            torch.save(model.state_dict(), args.save_pth + "_test_accuracy_" + str(test_accuracy) + ".pt")
        if best_test_balanced_accuracy < test_balanced_accuracy:
            torch.save(model.state_dict(), args.save_pth + "_test_balanced_accuracy_" + str(test_balanced_accuracy) + ".pt")

        epoch_time = end_time - start_time
        train_loss = running_loss / len(batched_trainset)
        train_accuracy = correct_predictions / total_predictions

        print("Epoch: ", epoch + 1, "/", args.epoch, " Train Loss: ", train_loss, " Accuracy: ", train_accuracy,
                
                " time spent: ", time.time() - start, " s")

    torch.save(model.state_dict(), args.save_pth)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'maxvit')
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train_features.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test_features.csv')
    parser.add_argument('--data_root', type = str, default = '/usr/scratch/abhimanyu/courses/ECE8803_FML/OLIVES')
    parser.add_argument('--data_aug', type =int, default = 1)
    parser.add_argument('--lr', type = float, default = 5e-4)
    parser.add_argument('--weight_decay', type = float, default = 0.1)
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--save_pth', type = str, default = '/storage/home/hpaceice1/abambhaniya3/DRSS-Severity-Classification-on-OCT-images/VIT_model_checkpoints/')
    parser.add_argument('--load_checkpoint', type = str, default = None)
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="the number of epochs to warmup (default: 5)")
    parser.add_argument("--lr-min", default=1e-5, type=float, help="minimum lr of lr schedule (default: 1e-5)")
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--lr_scheduler", type=bool, default = False, help="Wethear to turn of LR scheduling or not ") 
    parser.add_argument("--do_batch", type=int, default = 1, help="Wethear to do batching ") 
    
    return parser.parse_known_args()


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args, unkown = parse_args()

    print(f"Training {args.model} for {args.epoch} epochs. LR: {args.lr}, weight_decay = {args.weight_decay}, batch_size = {args.batch_size}. ")
    # Define transform
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_name = args.model + timestr + ".pth" 
    name = os.path.join(args.save_pth, base_name)
    args.save_pth = os.path.abspath(name)

    # # Define dataloader
    batched_trainset, batched_testset, train_freq, test_freq, train_meta_avg = dataloader.dataloader(args , args.model) 
    print("Len of Train:",len(batched_trainset)," , Len of Test dataset:",len(batched_testset))
    print(train_freq)
    print(test_freq)
    freq = np.array(train_freq) + np.array(test_freq)
    print(freq)
    # weight = freq / np.sum(freq)
    weight = [sum(freq) / (3 * count) for count in freq]

    weight = torch.tensor(weight, dtype=torch.float)
    print(weight)

    train_dnn(args, device, batched_trainset, batched_testset, weight, train_meta_avg, 3)


