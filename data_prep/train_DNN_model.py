import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

import torchvision.models 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from torchinfo import summary
import argparse
import os
import copy
import dataloader

# Define the model architecture
class ImageMetadataClassifier(nn.Module):
    def __init__(self):
        super(ImageMetadataClassifier, self).__init__()
#         vit_model = torchvision.models.vit_b_16(,dropout=0.2)
#         vit_model = torchvision.models.VisionTransformer(
#                 image_size = 224,
#                 patch_size=16,
#                 num_layers=4,
#                 num_heads=8,
#                 hidden_dim=512,
#                 mlp_dim=2048,
#                 dropout=0.2,
#                 num_classes = 32)
#         vit_model.conv_proj = nn.Conv2d(49 , 512 , kernel_size=(16,16), stride=(16,16))
#         vit_model.encoder.layers = nn.Sequential( *list(vit_model.encoder.layers.children()))      ## Have only 2 Encoders
#         vit_model.heads.head = nn.Linear(in_features=256, out_features=32, bias=True)
        
        mnasnet = torchvision.models.mnasnet1_0(num_classes = 32)
        self.image_features = nn.Sequential( nn.Conv2d(49 , 3 , kernel_size=(1,1), stride=(1,1)),
                mnasnet
        )
        self.metadata_fc = nn.Sequential(
                nn.Linear(5, 8),
                nn.ReLU(),
                # Add more layers as needed
                )
        self.classifier = nn.Linear(32 + 8, 3)

    def forward(self, x, metadata):
#         print(x.dtype,metadata.dtype)
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        metadata_extract = self.metadata_fc(metadata)
        x = torch.cat([x, metadata_extract], dim=1)
        x = self.classifier(x)
        return x


def train_dnn(args, device,batched_trainset, batched_testset ,num_class):
    # Define model
#     if( args.model.lower() == "maxvit"):
#         model = torchvision.models.maxvit_t()
#         model.classifier = nn.Sequential( nn.AdaptiveAvgPool2d(output_size=1),
#                                       nn.Flatten(start_dim=1, end_dim=-1),
#                                       nn.LayerNorm((512,), eps=1e-05, elementwise_affine=True),
#                                       nn.Linear(in_features=512, out_features=512, bias=True),
#                                       nn.Tanh(),
#                                       nn.Linear(in_features=512, out_features=num_class, bias=False))
# 
#     elif( args.model.lower() == "alexnet"): 
#         model = torchvision.models.alexnet()
#         model.classifier = nn.Sequential ( 
#                     nn.Dropout(p=0.5, inplace=False),
#                     nn.Linear(in_features=9216, out_features=4096, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout(p=0.5, inplace=False),
#                     nn.Linear(in_features=4096, out_features=4096, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(in_features=4096, out_features=num_class, bias=True),
#                     )
#     elif( args.model.lower()  == "vit"):
#         vit_model = torchvision.models.vit_b_16(dropout=0.5)
# #         model.conv_proj = nn.Conv2d(49, 768, kernel_size=(16, 16), stride=(16, 16))
#         vit_model.encoder.layers = nn.Sequential( *list(vit_model.encoder.layers.children())[0:2])      ## Have only 2 Encoders
#         vit_model.heads.head = nn.Linear(in_features=768, out_features=num_class, bias=True)
# #         model = nn.Sequential( nn.Conv2d(49 , 768 , kernel_size=(16,16), stride=(16,16) ))
#         vit_model.conv_proj = nn.Conv2d(49 , 768 , kernel_size=(16,16), stride=(16,16))
#         model = vit_model
#     else:
#         model = torchvision.models.resnet18()
    model = ImageMetadataClassifier()
    print(model)

#     print(summary(model, [1,49,224,224],[1,6]))
#     child_counter = 0
#     for child in model.encoder.layers.children():
#         print(" child", child_counter, "is -")
#         print(child)
#         child_counter += 1
    if( args.load_checkpoint is not None):
        try:
            print("Trying to Load Checkpoint")
            model.load_state_dict(torch.load(args.load_checkpoint))
            print("Load Checkpoint Successful")
        except:
            print("Checkpoint load failed")

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW( model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  
    ## Autoscaling while training can speedup train time on newer GPUs like V100 and A100
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ## Define LR schedular
    if(args.lr_scheduler):
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

#     metadata = torch.zeros((8,6)).to(device)
#     image = torch.zeros((8,49,224,224)).to(device)
#     out = model(image,metadata)
#     print(out)
    # Train model
    num_epochs = args.epoch
    start = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0
        start_time = time.time()
        model.train()
        # Train 1 epoch
        for images, labels, metadata in tqdm(batched_trainset, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data and labels to device
            images = images.to(device)
            labels = labels.to(device)
            metadata = metadata.to(device)
            images = torch.flatten(images , start_dim = 1 , end_dim =2)
#             print(metadata.shape, metadata)
            # Forward pass
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(x = images, metadata = metadata)
#                 print(outputs)
                loss = criterion(outputs, labels)


            # Backward pass
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)       ## 1.0 is args.clip_grad_norm
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)       ## 1.0 is args.clip_grad_norm
                optimizer.step()

            # Record training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
   
        if(args.lr_scheduler):
            lr_scheduler.step()

        if epoch % 3 == 0:
            # Evaluate model on test data
            test_loss = 0.0
            test_correct_predictions = 0.0
            test_total_predictions = 0.0
    
            model.eval()
            with torch.no_grad():
                for images, labels, _ in batched_testset:
                    images = images.to(device)
                    labels = labels.to(device)
                    metadata = torch.zeros(_.shape, device= device)

                    images = torch.flatten(images , start_dim = 1 , end_dim =2)
                    # Forward pass
                    outputs = model(images,metadata)
                    loss = criterion(outputs, labels)
    
                    # Record test loss and accuracy
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total_predictions += labels.size(0)
                    test_correct_predictions += (predicted == labels).sum().item()
#                     print(" predicted :", predicted)
#                     print(" labels :", labels)
            test_loss /= len(batched_testset)
            test_accuracy = test_correct_predictions / test_total_predictions
            print(" Test Loss: ", test_loss, " Accuracy: ", test_accuracy)

        # Print statistics and add to Tensorboard
        end_time = time.time()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.save_pth + "epoch" + str(epoch) + ".pt")
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
    parser.add_argument('--lr', type = float, default = 3e-3)
    parser.add_argument('--weight_decay', type = float, default = 0.05)
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
    batched_trainset, batched_testset = dataloader.dataloader(args , args.model) 
    print("Len of Train:",len(batched_trainset)," , Len of Test dataset:",len(batched_testset))
    train_dnn(args, device, batched_trainset, batched_testset, 3)


