from collections import defaultdict
import functools
import pathlib
from PIL import Image
import io
import json
import os

import numpy as np
from scipy.special import softmax
import torch

import clip
import ImageReward as RM
from lavis.models import load_model_and_preprocess
from transformers import AutoModel, AutoProcessor


###############
# Base scores #
###############
def scale_norm(l, c=1.0):
    return (l - np.min(l)) / (np.max(l) - np.min(l)) * c


def softmax_score(logits, scale=1.0):
    logits = logits * scale
    return softmax(logits, axis=-1)


def mean_ensemble_score(logits_l, scale):
    ensemble_probs = np.concatenate([
        softmax_score(scale_norm(logits, scale))[:, [0]]
        for logits in logits_l
    ], axis=1)
    mean_scores = np.mean(ensemble_probs, axis=1)
    return mean_scores, ensemble_probs


def pickscore_base(images, prompts, device):
    images = (images * 255).round().clamp(0, 255).to(torch.uint8)

    # Preprocess.
    # Can be either a list of PIL.Image objects or a batched np.array.
    image_inputs = pickscore_processor(
        images=images, padding=True, truncation=True, max_length=77,
        return_tensors='pt').to(device)
    text_inputs = pickscore_processor(
        text=prompts, padding=True, truncation=True, max_length=77,
        return_tensors='pt').to(device)

    with torch.no_grad():
        # Embeddings: [batch_size, 1024].
        image_embs = pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Pick score.
        embs_dot = torch.sum(torch.multiply(text_embs, image_embs), dim=-1)
        scores = pickscore_model.logit_scale.exp() * embs_dot / 100.0

    return scores.cpu().numpy()


def pickscore_base2(image, prompt, base_prompts, device):
    base_prompts = set(base_prompts) - set([prompt])
    base_prompts = [prompt] + sorted(list(base_prompts))

    # Preprocess.
    # Can be either a list of PIL.Image objects or a batched np.array.
    image_inputs = pickscore_processor(
        images=[image], padding=True, truncation=True, max_length=77,
        return_tensors='pt').to(device)
    text_inputs = pickscore_processor(
        text=base_prompts, padding=True, truncation=True, max_length=77,
        return_tensors='pt').to(device)

    with torch.no_grad():
        # Embeddings: [batch_size, 1024].
        image_embs = pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        text_embs = pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        logits = image_embs @ text_embs.t()
        logits = pickscore_model.logit_scale.exp() * logits

    logits = logits.cpu().numpy()
    return logits


def imagereward_base(images, prompts, device):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]
    rewards = []
    for caption, image in zip(prompts, images):
        rewards.append(ir_model.score(caption, image))
    return np.array(rewards)


def imagereward_base2(image, prompt, base_prompts, device):
    base_prompts = set(base_prompts) - set([prompt])
    base_prompts = [prompt] + sorted(list(base_prompts))
    logits = []  # To keep shapes consistent.
    logits.append(ir_model.score(base_prompts, [image]))
    return logits


def clip_score_base(images, prompts, device):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    image_inputs = [
        clip_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        for image in images
    ]
    image_inputs = torch.concat(image_inputs, axis=0)
    text_inputs = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        # Cosine sim between the image and text features times 100.
        # Shape: [batch, batch].
        logits_per_image, _ = clip_model(image_inputs, text_inputs)
        # Only take the scores for pairwise image and text features.
        logits = torch.diagonal(logits_per_image, 0)

    return logits.cpu().numpy() / 100.0


def blip_score_base(images, prompts, device):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    image_inputs = [
        blip_vis_processor['eval'](
            Image.fromarray(image).convert('RGB')).unsqueeze(0).to(device)
        for image in images
    ]
    image_inputs = torch.concat(image_inputs, axis=0)
    text_inputs = [blip_txt_processor['eval'](prompt) for prompt in prompts]
    sample = {'image': image_inputs, 'text_input': text_inputs}

    features_image = blip_model.extract_features(sample, mode='image')
    features_text = blip_model.extract_features(sample, mode='text')

    # Use low-dimensional projected features for similarity scoring.
    image_embeds_proj = features_image.image_embeds_proj  # [batch, 32, 256]
    text_embeds_proj = features_text.text_embeds_proj     # [batch, td, 256]
    similarity = image_embeds_proj @ text_embeds_proj[:, 0, :].unsqueeze(-1)
    similarity = torch.max(similarity, dim=1).values.squeeze(-1)
    return similarity.cpu().numpy()


###################
# Score functions #
###################
def pickscore():

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('pickscore', device)
        scores = pickscore_base(images, prompts, device)
        return scores, {}

    return _fn


def imagereward():

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('imagereward', device)
        scores = imagereward_base(images, prompts, device)
        return scores, {}

    return _fn


def clip_score():

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('clip', device)
        scores = clip_score_base(images, prompts, device)
        return scores, {}

    return _fn


def blip_score():

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('blip', device)
        scores = blip_score_base(images, prompts, device)
        return scores, {}

    return _fn


def read_base_prompt_dict(prompt_type):
    if prompt_type.startswith('compose'):
        prompt_jsons = ['data_llm1.6_composition.json',
                        'data_manual_composition.json']
    elif prompt_type.startswith('count'):
        prompt_jsons = ['data_manual_counting.json']
    elif prompt_type.startswith('open100'):
        prompt_jsons = ['data_llm_open100.json']
    else:
        raise ValueError(f'Unsupported prompt type: {prompt_type}')

    filedir = pathlib.Path(__file__).parent.resolve()
    base_prompt_dict = defaultdict(list)
    for prompt_json in prompt_jsons:
        with open(os.path.join(filedir, prompt_json), 'r') as f:
            for prompt, base_prompts in json.load(f).items():
                base_prompt_dict[prompt].extend(base_prompts)
    return base_prompt_dict


def read_metadata_dict(prompt_type):
    if prompt_type.startswith('compose'):
        filename = 'data_meta_composition.json'
    elif prompt_type.startswith('count'):
        filename = 'data_meta_counting.json'
    elif prompt_type.startswith('open100'):
        filename = 'data_meta_open100.json'
    else:
        raise ValueError(f'Unsupported prompt type: {prompt_type}')

    filedir = pathlib.Path(__file__).parent.resolve()
    with open(os.path.join(filedir, filename), 'r') as f:
        return json.load(f)


def mean_ensemble(prompt_type):
    base_prompt_dict = read_base_prompt_dict(prompt_type)
    metadata_dict = read_metadata_dict(prompt_type)

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('mean_ensemble', device)

        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]

        scores = []
        for image, prompt in zip(images, prompts):
            base_prompts = base_prompt_dict[prompt]
            ir_logits = imagereward_base2(image, prompt, base_prompts, device)
            ps_logits = pickscore_base2(image, prompt, base_prompts, device)
            scale = metadata_dict[prompt]['mean_best']['scale']
            ensemble_scores, _ = mean_ensemble_score([ir_logits, ps_logits], scale)
            assert(len(ensemble_scores) == 1)
            scores.append(ensemble_scores[0])
        return scores, {}

    return _fn


def uw_ensemble(prompt_type):
    base_prompt_dict = read_base_prompt_dict(prompt_type)
    metadata_dict = read_metadata_dict(prompt_type)

    def _fn(images, prompts, metadata):
        del metadata
        device = images.device
        init_reward_model('mean_ensemble', device)

        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]

        scores = []
        for image, prompt in zip(images, prompts):
            base_prompts = base_prompt_dict[prompt]
            ir_logits = imagereward_base2(image, prompt, base_prompts, device)
            ps_logits = pickscore_base2(image, prompt, base_prompts, device)
            scale = metadata_dict[prompt]['uw_best']['scale']
            lmbda = metadata_dict[prompt]['uw_best']['lambda']
            ensemble_scores, ensemble_probs = mean_ensemble_score([ir_logits, ps_logits], scale)
            assert(len(ensemble_scores) == 1 and len(ensemble_probs) == 1)
            mean_score, probs = ensemble_scores[0], ensemble_probs[0]
            variance = np.mean((probs - mean_score) ** 2.0)
            uw_score = mean_score - lmbda * variance
            scores.append(uw_score)
        return scores, {}

    return _fn


clip_model, clip_preprocess = None, None
blip_model, blip_vis_processor, blip_txt_processor = None, None, None
pickscore_model, pickscore_processor = None, None
ir_model = None


def init_reward_model(reward_type, device):
    if reward_type not in ('clip', 'blip', 'imagereward', 'pickscore',
                           'mean_ensemble', 'uw_ensemble'):
        raise ValueError(f'Unsupported reward type: {reward_type}')

    if reward_type == 'clip':
        global clip_model, clip_preprocess
        if clip_model is None:
            clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
            clip_model.eval()
    if reward_type == 'blip':
        global blip_model, blip_vis_processor, blip_txt_processor
        if blip_model is None:
            blips = load_model_and_preprocess(name='blip2_feature_extractor',
                                            model_type='pretrain', is_eval=True,
                                            device=device)
            blip_model, blip_vis_processor, blip_txt_processor = blips
    if reward_type in ('imagereward', 'mean_ensemble', 'uw_ensemble'):
        global ir_model
        if ir_model is None:
            ir_model = RM.load('ImageReward-v1.0', device=device)
    if reward_type in ('pickscore', 'mean_ensemble', 'uw_ensemble'):
        global pickscore_processor, pickscore_model
        if pickscore_model is None:
            processor_name_or_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
            model_pretrained_name_or_path = 'yuvalkirstain/PickScore_v1'
            pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
            pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path)
            pickscore_model.eval().to(device)


def get_reward_model(reward_type, prompt_type=None):
    if reward_type == 'clip': return clip_score()
    if reward_type == 'blip': return blip_score()
    if reward_type == 'imagereward': return imagereward()
    if reward_type == 'pickscore': return pickscore()
    if reward_type == 'mean_ensemble': return mean_ensemble(prompt_type)
    if reward_type == 'uw_ensemble': return uw_ensemble(prompt_type)
    raise ValueError(f'Unsupported reward type: {reward_type}')


if __name__ == '__main__':
    image_fn = '/data/jongheon_jeong/dev/trl/analysis/deer_truck.png'
    image = Image.open(image_fn)
    testcase = 'ensemble'

    device = 'cuda'

    if testcase == 'clip':
        get_reward_model('clip')
        print(clip_preprocess(image).shape)
        img_arr = np.asarray(image)
        print(img_arr.shape)
        print(clip_preprocess(Image.fromarray(img_arr)).shape)
    if testcase == 'pickscore':
        print(np.asarray(image).shape)
        get_reward_model('pickscore')
        image_inputs = pickscore_processor(
            images=[image], padding=True, truncation=True, max_length=77,
            return_tensors='pt').to(device)
        image_embs1 = pickscore_model.get_image_features(**image_inputs).cpu().detach().numpy()
        image_inputs = pickscore_processor(
            images=np.expand_dims(np.asarray(image), 0), padding=True, truncation=True, max_length=77,
            return_tensors='pt').to(device)
        image_embs2 = pickscore_model.get_image_features(**image_inputs).cpu().detach().numpy()
        print(np.allclose(image_embs1, image_embs2))
    if testcase == 'ensemble':
        init_reward_model('uw_ensemble', device)
        get_reward_model('uw_ensemble', 'compose_10')
        prompt = 'a deer and a truck'
        base_prompts = ['a deer', 'a truck', 'a tree']
        ir_logits = imagereward_base2(image, prompt, base_prompts, device)
        ps_logits = pickscore_base2(image, prompt, base_prompts, device)
        ensemble_scores, ensemble_probs = mean_ensemble_score([ir_logits, ps_logits], 5.0)
        assert(len(ensemble_scores) == 1 and len(ensemble_probs) == 1)
        mean_score, probs = ensemble_scores[0], ensemble_probs[0]
        variance = np.mean((probs - mean_score) ** 2.0)
        uw_score = mean_score - 2.0 * variance
        print(uw_score)
