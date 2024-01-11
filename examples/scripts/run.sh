#!/bin/bash

device=0
num_seeds=2
num_imgs_per_seed=10
# reward_type=
# reward_type=imagereward
reward_type=imagereward
prompt_type=subcomposition

_basedir=/data/jongheon_jeong/dev/trl
pyfile=${_basedir}/examples/scripts/ddpo.py
# outdir=${_basedir}/save/${prompt_type}/${reward_type}/b64_lr0.0003

source /data/jongheon_jeong/anaconda3/bin/activate ddpo

CUDA_VISIBLE_DEVICES=${device} accelerate launch \
--main_process_port=1334 --num_processes=1 --mixed_precision=bf16 ${pyfile} \
--num_seeds=${num_seeds} --num_imgs_per_seed=${num_imgs_per_seed}

# accelerate launch --num_processes=1 --main_process_port=13340 examples/scripts/ddpo.py
