#!/bin/bash
# This is a bash file that provide a execution for resnet training from my local machine, pls modify to fit yours
python train_svm.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --save_pickle ../../../ece8803/models/ --log ../../../ece8803/models/ --n_components 49