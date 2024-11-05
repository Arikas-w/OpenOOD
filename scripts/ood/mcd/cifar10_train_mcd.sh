#!/bin/bash
# sh scripts/ood/mcd/cifar10_train_mcd.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_oe.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mcd.yml \
    --network.backbone.name resnet18_32x32 \
    --network.pretrained False \
    --dataset.image_size 32 \
    --optimizer.num_epochs 100 \
    --num_workers 8 \
    --seed ${SEED}
