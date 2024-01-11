#!/bin/bash

num_seeds=1
num_imgs_per_seed=10

_basedir=/data/jongheon_jeong/dev/trl
pyfile=${_basedir}/examples/scripts/generate.py

source /data/jongheon_jeong/anaconda3/bin/activate ddpo

i=0
port=2230
for prompt_type in open100_10; do
    for rtype in uw_ensemble; do
        outdir_base=${_basedir}/save/${prompt_type}/${rtype}

        for d in $(ls -d ${outdir_base}/b64_*); do
            dir_base=$(basename ${d})
            device=$(( i % 1 ))

            echo CUDA_VISIBLE_DEVICES=${device} accelerate launch --num_processes=1 --main_process_port=${port} ${pyfile} \
            --prompt_type=${prompt_type} \
            --num_seeds=${num_seeds} --num_imgs_per_seed=${num_imgs_per_seed} \
            --outdir=${d} --lora_paths=all --reward_type=${rtype} &
            (( ++i ))
            (( ++port ))
            if (( i % 1 == 0 )); then
                wait
                echo "Waited ${i}"
            fi
            break
        done
    done
done


# CUDA_VISIBLE_DEVICES=${device} accelerate launch --main_process_port=1334 --num_processes=1 ${pyfile} \
# --num_seeds=${num_seeds} --num_imgs_per_seed=${num_imgs_per_seed} \
# --outdir=${outdir} --reward_type=${reward_type} --lora_paths=all \
# --prompt_type=${prompt_type}
