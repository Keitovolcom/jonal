#!/usr/bin/env bash
set -euo pipefail

# ResNet-18 (CIFAR-style stem) on MNIST 28x28, SGD without momentum/weight decay.
#   LR=1e-5 GPU=0 bash train_mnist_resnet18.bash
#   LR=1e-2 GPU=1 bash train_mnist_resnet18.bash
LR="${LR:?set LR, e.g. LR=1e-2}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
INIT="${INIT:-kaiming}"  # kaiming | default (PyTorch layer defaults)
WANDB_PROJECT="${WANDB_PROJECT:-kobayashi_syuron}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-dsml-kernel24}"

python3 train.py \
  --model resnet18 \
  --dataset mnist \
  --input-channels 1 \
  --lr "${LR}" \
  --batch-size 128 \
  --gpu "${GPU}" \
  --label-noise 0.2 \
  --optimizer sgd \
  --momentum 0.0 \
  --weight-decay 0.0 \
  --loss crossentropy \
  --augmentation none \
  --epochs "${EPOCHS}" \
  --seed 42 \
  --init "${INIT}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-entity "${WANDB_ENTITY}"
