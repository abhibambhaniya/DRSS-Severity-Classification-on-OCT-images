#!/bin/bash
# run with Adam
python train_resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --epoch 40 --batch_size 16 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/ --save_pred ../../../ece8803/models/ --lr 0.00001 --weight_decay 0.05