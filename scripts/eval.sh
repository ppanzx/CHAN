## f30k
DATASET_NAME='f30k'
DATA_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/'${DATASET_NAME}
VOCAB_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/vocab'
SAVE_PATH='checkpoints/f30k_gru'

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results

## coco
DATASET_NAME='coco'
DATA_PATH='/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval/'${DATASET_NAME}
SAVE_PATH="checkpoints/coco_bert"

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth \
    --data_path=${DATA_PATH} --save_results