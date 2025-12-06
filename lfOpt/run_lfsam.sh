#!/usr/bin/env bash

# Run training with Low-Frequency SAM (SAM + LowFreqAdamW base) on CIFAR100 / ResNet18.

device=0
seed=1
datasets=CIFAR100
model=resnet18 # resnet18 VGG16BN WideResNet28x10
schedule=cosine
wd=0.001
epoch=200
bz=128
rho=0.2
sigma=1     # for naming consistency
lmbda=0.6   # for naming consistency
opt=LFSAM

# LowFreqAdam-specific hyperparameters
lf_m=4
lf_sigma=1.0
lf_lam=0.5
lf_scale_match=false
lf_base=adam     # adam | sgd

DST=results/$opt/$datasets/$model/${opt}_cutout_${rho}_${sigma}_${lmbda}_${epoch}_${model}_bz${bz}_wd${wd}_${datasets}_${schedule}_seed${seed}

CUDA_VISIBLE_DEVICES=$device python -u train.py --datasets $datasets \
        --arch=$model --epochs=$epoch --wd=$wd --randomseed $seed --lr 0.05 --rho $rho --optimizer $opt \
        --save-dir=$DST/checkpoints --log-dir=$DST -p 200 --schedule $schedule -b $bz \
        --cutout --sigma $sigma --lmbda $lmbda \
        --lf_m $lf_m --lf_sigma $lf_sigma --lf_lam $lf_lam --lf_base $lf_base \
        $( [ "$lf_scale_match" = true ] && echo "--lf_scale_match" ) \
        --wandb
