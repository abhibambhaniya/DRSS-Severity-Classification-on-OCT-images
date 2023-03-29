#!/bin/bash
# run with Adam
python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.001 --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/

python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.01 --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/

python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.1 --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/

# run with SGD
python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.001 --opt SGD --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/

python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.01 --opt SGD --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/

python resnet.py --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml/ --lr 0.1 --opt SGD --epoch 50 --save_pth ../../../ece8803/models/ --log ../../../ece8803/models/



