# Copyright 2023 metric-space, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import tyro

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline

import prompts
import reward_fns


@dataclass
class ScriptArguments:
    hf_user_access_token: str = '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'
    # 'stabilityai/stable-diffusion-2-1' or 'stabilityai/stable-diffusion-2-1-base'
    pretrained_model: str = 'stabilityai/stable-diffusion-2-1-base'  # 'runwayml/stable-diffusion-v1-5'
    """the pretrained model to use"""
    pretrained_revision: str = "main"
    """the pretrained model revision to use"""
    hf_hub_model_id: str = "ddpo-finetuned-stable-diffusion"
    """HuggingFace repo to save model weights to"""
    hf_hub_aesthetic_model_id: str = "trl-lib/ddpo-aesthetic-predictor"
    """HuggingFace model ID for aesthetic scorer model weights"""
    hf_hub_aesthetic_model_filename: str = "aesthetic-model.pth"
    """HuggingFace model filename for aesthetic scorer model weights"""

    ddpo_config: DDPOConfig = field(
        default_factory=lambda: DDPOConfig(
            resume_from=resume_from,
            num_epochs=num_epochs,
            save_freq=10,
            num_checkpoint_limit=10000,
            # train_gradient_accumulation_steps=1,
            train_gradient_accumulation_steps=train_gradient_accumulation_steps,
            sample_num_steps=50,
            # sample_batch_size=6,
            sample_batch_size=sample_batch_size,
            # train_batch_size=3,
            train_batch_size=train_batch_size,
            # sample_num_batches_per_epoch=4,
            train_learning_rate=train_learning_rate,
            sample_num_batches_per_epoch=sample_num_batches_per_epoch,
            per_prompt_stat_tracking=True,
            per_prompt_stat_tracking_buffer_size=32,
            tracker_project_name="stable_diffusion_training",
            log_with='tensorboard',
            project_kwargs={
                "logging_dir": "./logs",
                "automatic_checkpoint_naming": True,
                "total_limit": 100000,
                # "project_dir": "./save",
                "project_dir": project_dir,
            },
        )
    )


if __name__ == '__main__':
    resume_from = ''
    # resume_from = './save/composition5/clip/b64_lr0.0003/checkpoints'

    # Config overrides.
    num_epochs = 1000
    # The DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    num_gpus = torch.cuda.device_count()
    multiplier = int(8 / num_gpus)
    sample_batch_size = 16  # Over 16 doesn't seem to give much more speed-up.
    sample_num_batches_per_epoch = 2 * multiplier
    # This corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    train_batch_size = 8
    train_gradient_accumulation_steps = 1 * multiplier
    train_learning_rate = 3e-4

    # Type of reward: 'clip', 'blip', 'imagereward', 'pickscore', 'mean_ensemble', 'uw_ensemble'
    reward_type = 'uw_ensemble'
    # Type of prompt: 'composition', 'counting', 'open100'.
    limit = None
    prompt_type = 'open100_10'
    prompt_dir = f'{prompt_type}-l{limit}' if limit else prompt_type
    project_dir = f'./save/{prompt_dir}/{reward_type}'

    args = tyro.cli(ScriptArguments)

    # Initialize the needed reward models.
    scorer = reward_fns.get_reward_model(reward_type, prompt_type)
    train_prompts = prompts.get_prompts(prompt_type)
    train_prompts = train_prompts[:limit] if limit else train_prompts
    def prompt_fn():
        return np.random.choice(train_prompts), {}

    use_lora = True
    pipeline = DefaultDDPOStableDiffusionPipeline(
        args.pretrained_model, pretrained_model_revision=args.pretrained_revision, use_lora=use_lora
    )

    trainer = DDPOTrainer(
        args.ddpo_config,
        scorer,
        prompt_fn,
        pipeline,
    )

    trainer.train()
