#!/bin/bash
CFG=configs/SkyTimelapse/sky_ldmae_v1.yaml
# Debug
CUDA_VISIBLE_DEVICES=0 python main.py --logdir experiments/ --base $CFG  --debug True --ngpu 1
#--ckpt
