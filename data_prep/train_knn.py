import torch
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

import pandas as pd
from PIL import Image
import argparse
import os
import copy

import numpy as np

def knn_dataloader(args, file = 'df_prime_train_features.csv'):
    LABELS_Severity = {35: 0,
                    43: 0,
                    47: 1,
                    53: 1,
                    61: 2,
                    65: 2,
                    71: 2,
                    85: 2}


    annot = pd.read_csv(file)
    annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(annot['DRSS'].values)]
    # root = os.path.expanduser(args.data_root)
    # nb_classes=len(np.unique(list(LABELS_Severity.values())))
    path_list = annot['Volume_ID'].values

    labels = annot['Severity_Label'].values.reshape(-1,1)
    # print(labels)
    # assert len(path_list) == len(labels)
    root = os.path.expanduser(args.data_root)


    img_volume = []

    # for index in range(20):
    for index in range(len(path_list)):
        # img_volume[index] = []
        frames = []
        folder_path = root + path_list[index]

        # if index%10 == 0  :
            # print(f"Completed {index} images")
        
        starting_frame = 10
        for i in range(10, 39): 
            tif = str(i) + '.tif'
            png = str(i) + '.png'
            
            if (os.path.isfile(os.path.join(folder_path, tif))):
                img = Image.open(os.path.join(folder_path, tif)).convert("L")
            elif (os.path.isfile(os.path.join(folder_path, png))):
                img = Image.open(os.path.join(folder_path, png)).convert("L")
            else:
                img = frames[i -starting_frame - 1]
                frames.append(frames[i -starting_frame - 1])
                continue
            frames.append(np.asarray(img)[106:224,80:450])


        img_volume.append(np.array(frames))

    print(np.shape(img_volume))
    img_volume =np.reshape(img_volume,(len(img_volume), -1)) 
    # ret = img_volume.reshape((len(img_volume), -1))
    print(np.shape(labels))

    return img_volume, labels

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
    parser.add_argument('--batch_size', type = int, default = 1000)
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


    # Load your 3D image dataset
    # Assuming your dataset is loaded into a PyTorch tensor called `data` with shape (num_samples, num_features)
    # and the corresponding labels are loaded into a PyTorch tensor called `labels` with shape (num_samples,)

    # Split the dataset into training and testing sets
    X_train, y_train = knn_dataloader(args, args.annot_train_prime )
    X_test, y_test = knn_dataloader(args, args.annot_test_prime ) 
    # Create and train KNN classifier
    
    y_pred_all = []
    for k in [3,5,7,9,11,15]:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        # Predict on test data
        y_pred = clf.predict(X_test)
        y_pred_all.append(y_pred)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")