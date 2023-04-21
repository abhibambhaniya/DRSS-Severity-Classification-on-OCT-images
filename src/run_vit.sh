python train_DNN_model.py --model vit\
    --epoch 30 --batch_size 8 --lr 5e-3 --weight-decay 0.1 --lr-min 1e-4 --lr-warmup-epochs 5 --lr_scheduler True\
    --amp --data_root /storage/home/hpaceice1/shared-classes/materials/ece8803fml \
    --load_checkpoint /storage/home/hpaceice1/abambhaniya3/DRSS-Severity-Classification-on-OCT-images/VIT_model_checkpoints/vit20230413-214119_test_balanced_accuracy_.pt 
