#!/bin/bash

# custom config
TRAINER=LocProto

CSC=False
CTP=end

DATA=./datasets/
# DATASET=skin40
CFG=vit_b16_ep25

NCTX=16

T=1
# SHOTS=16
# MODEL_dir=$5
# Output_dir=$5

# SEED=2
for DATASET in skin40  # eurosat fgvc_aircraft stanford_cars skin40 ISIC
do
    for SHOTS in 16
    do
        for SEED in 1
        do
            CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py \
            --root ${DATA} \
            --in_dataset ${DATASET} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --seed ${SEED} \
            --output-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa \
            --model-dir output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}_aaaaaa \
            --load-epoch 200 \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --T ${T} \
            --is_bonder True \
            --is_dense True \
            DATASET.SUBSAMPLE_CLASSES base \
            DATASET.NUM_SHOTS ${SHOTS}
        done
    done
done