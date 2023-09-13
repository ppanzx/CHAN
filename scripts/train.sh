export CUDA_VISIBLE_DEVICES=0

## f30k
DATASET_NAME="f30k"
USER_NAME="/home/panzx"
DATASET_ROOT="${USER_NAME}/dataset/CrossModalRetrieval"
DATA_PATH="${DATASET_ROOT}/${DATASET_NAME}"
VOCAB_PATH="${DATASET_ROOT}/vocab"

SAVE_PATH='checkpoints/f30k_gru'
python ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=selfattention --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=384 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05 

python eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}

SAVE_PATH='checkpoints/f30k_bert'
python ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bert \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=MeanPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 

python eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}

## coco
DATASET_NAME='coco'
DATASET_ROOT="${USER_NAME}/dataset/CrossModalRetrieval"
DATA_PATH="${DATASET_ROOT}/${DATASET_NAME}"

SAVE_PATH='checkpoints/coco_gru'
python ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=selfattention --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=384 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05 

python eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}


SAVE_PATH='checkpoints/coco_bert'
python ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bert \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=MeanPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 

python eval.py \
  --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}

  