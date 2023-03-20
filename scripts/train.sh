## f30k
DATASET_NAME='f30k'
DATA_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/'${DATASET_NAME}
VOCAB_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/vocab'
SAVE_PATH='checkpoints/f30k_gru'

CUDA_VISIBLE_DEVICES=0 python3 ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=selfattention --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=384 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05 \

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}

## coco
DATASET_NAME='coco'
DATA_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/'${DATASET_NAME}
SAVE_PATH='checkpoints/coco_bert'
CUDA_VISIBLE_DEVICES=0 python3 ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bert \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 \

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}