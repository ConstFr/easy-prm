#!/usr/bin/env bash

# NUM_GPUS=${NUM_GPUS:-4}
NUM_GPUS=4

torchrun --nproc_per_node="${NUM_GPUS}" train_script.py -c train_configs/deberta_prm.yml "$@"
