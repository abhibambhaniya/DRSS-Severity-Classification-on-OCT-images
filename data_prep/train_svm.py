# Refer to https://github.com/kenshohara/3D-ResNets-PyTorch 
# Get pretrained model from: https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


import dataloader

import pandas as pd
from PIL import Image
import argparse
import os
import copy
import time
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train_features.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test_features.csv')
    parser.add_argument('--data_root', type = str, default = '')
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--epoch', type = int, default = 50)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--log', type = str, default = '/usr/scratch/yangyu/FML_Model/SVM')
    parser.add_argument('--save_pickle', type = str, default = '/usr/scratch/yangyu/FML_Model/SVM')

    return parser.parse_args()


## Take an hour to finish
## Train Feature Size: (495, 980) Test Feature Size: (163, 980)
## Accuracy: 0.4049079754601227

if __name__ == '__main__':
    args = parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    base_name = "svm_" + timestr + ".pickle"
    name = os.path.join(args.save_pickle, base_name)

    feature_base_name = "svm_feature_" + timestr + ".pickle"
    save_feature = os.path.abspath(os.path.join(args.save_pickle, feature_base_name))

    args.save_pickle = os.path.abspath(name)

    print(args.save_pickle)
    print(save_feature)
    
    base_name_log = "svm_" + timestr + ".log"
    name_log = os.path.join(args.log, base_name_log)
    args.log = os.path.abspath(name_log)
    
    start_time = time.time()
    train_features, train_labels, test_features, test_labels = dataloader.svm_dataloader(args, 'SVM')

    dict = {"train_features": train_features, "train_labels": train_labels, "test_features": test_features, "test_labels" : test_labels}
    with open(save_feature,'wb') as f:
        pickle.dump(dict, f)


    logfile = open(args.log, "w")
    msg = f'Train Feature Size: {np.shape(train_features)}, Test Feature Size: {np.shape(test_features)} \n'

    # print("Train Feature Size:", np.shape(train_features), "Test Feature Size:", np.shape(test_features))
    logfile.write(msg)

    svm = SVC(kernel='linear')
    svm.fit(train_features, train_labels)

    # Predict the labels of the testing set using the SVM
    test_predictions = svm.predict(test_features)

    # Evaluate the performance of the SVM using metrics such as accuracy, precision, recall, or F1 score
    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)

    accuracy_msg = f'Fit and Predition time: {time.time() - start_time}, Accuracy: {accuracy}, Balanced Accuracy: {balanced_accuracy}'
    
    logfile.write(accuracy_msg)
    logfile.write('\n')

    print(accuracy_msg)

    logfile.close()

    with open(args.save_pickle,'wb') as f:
        pickle.dump(svm, f)


    # with open('model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    #     clf2.predict(test_features)
