#!/usr/bin/env bash

# Simple wrapper to run SAMPa experiments from within the F-SAM repo.

device=0
seed=42
dataset=cifar10
model=resnet56
epochs=200
batch_size=128
lr=0.1
rho=0.1      # \\rho parameter for SAMPa
og=0.2       # \\lambda parameter for SAMPa
wd=0.0005
momentum=0.9
save_path=results/SAMPa/${dataset}/${model}

mkdir -p "${save_path}"

CUDA_VISIBLE_DEVICES=${device} python3 ../SAMPa/train.py \
  --dataset ${dataset} \
  --model ${model} \
  --epochs ${epochs} \
  --batch-size ${batch_size} \
  --lr ${lr} \
  --rho ${rho} \
  --og ${og} \
  --weight_decay ${wd} \
  --momentum ${momentum} \
  --seed ${seed} \
  --save_path "${save_path}"

