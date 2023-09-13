export CUDA_VISIBLE_DEVICES=0

## f30k
DATASET_NAME="f30k"
USER_NAME="/home/panzx"
DATASET_ROOT="${USER_NAME}/dataset/CrossModalRetrieval"
DATA_PATH="${DATASET_ROOT}/${DATASET_NAME}"
VOCAB_PATH="${DATASET_ROOT}/vocab"

SAVE_PATH='checkpoints/f30k_gru'
python eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results

SAVE_PATH='checkpoints/f30k_bert'
python eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results

## coco
DATASET_NAME="coco"
DATA_PATH="${DATASET_ROOT}/${DATASET_NAME}"

SAVE_PATH="checkpoints/coco_gru"
python eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results

SAVE_PATH="checkpoints/coco_bert"
python eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results