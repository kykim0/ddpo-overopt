"""Utils to compute various rewards."""

from PIL import Image

import clip
import ImageReward as RM
from lavis.models import load_model_and_preprocess
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import reward_fns


clip_model, clip_preprocess = None, None  # CLIP.
blip_model, blip_vis_processor, blip_txt_processor = None, None, None
pickscore_model, pickscore_processor = None, None  # PickScore.
ir_model = None  # ImageReward.

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _init_reward_models(reward_type):
  if reward_type == 'clip':
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()
  elif reward_type == 'blip':
    global blip_model, blip_vis_processor, blip_txt_processor
    blips = load_model_and_preprocess(name='blip2_feature_extractor',
                                      model_type='pretrain', is_eval=True,
                                      device=device)
    blip_model, blip_vis_processor, blip_txt_processor = blips
  elif reward_type == 'imagereward':
    global ir_model
    ir_model = RM.load('ImageReward-v1.0')
  elif reward_type == 'pickscore':
    global pickscore_model, pickscore_processor
    processor_name_or_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    model_pretrained_name_or_path = 'yuvalkirstain/PickScore_v1'
    pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path)
    pickscore_model.eval().to(device)
  else:
    raise ValueError(f'Unsupported reward type: {reward_type}')


def clip_score(captions, image_paths):
  global clip_model
  if not clip_model: _init_reward_models('clip')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  image_inputs = [
      clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
      for image_path in image_paths
  ]
  image_inputs = torch.concat(image_inputs, axis=0)
  text_inputs = clip.tokenize(captions).to(device)

  with torch.no_grad():
    # Cosine sim between the image and text features times 100.
    # Shape: [batch, batch].
    logits_per_image, _ = clip_model(image_inputs, text_inputs)
    # Only take the scores for pairwise image and text features.
    logits = torch.diagonal(logits_per_image, 0)

    # Cosine sim can also be computed as follows:
    #   image_features = clip_model.encode_image(image_inputs)
    #   text_features = clip_model.encode_text(text_inputs)
    #   image_features /= image_features.norm(dim=-1, keepdim=True)
    #   text_features /= text_features.norm(dim=-1, keepdim=True)
    #   similarity = image_features @ text_features.T
    #   similarity = torch.diagonal(similarity, 0)

  return (logits.cpu().numpy() / 100.0).tolist()


def blip_score(captions, image_paths):
  global blip_model
  if not blip_model: _init_reward_models('blip')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  image_inputs = [
      blip_vis_processor['eval'](
          Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
      for image_path in image_paths
  ]

  image_inputs = torch.concat(image_inputs, axis=0)
  text_inputs = [blip_txt_processor['eval'](caption) for caption in captions]
  sample = {'image': image_inputs, 'text_input': text_inputs}

  features_image = blip_model.extract_features(sample, mode='image')
  features_text = blip_model.extract_features(sample, mode='text')

  # Use low-dimensional projected features for similarity scoring.
  image_embeds_proj = features_image.image_embeds_proj  # [batch, 32, 256]
  text_embeds_proj = features_text.text_embeds_proj     # [batch, td, 256]
  similarity = image_embeds_proj @ text_embeds_proj[:, 0, :].unsqueeze(-1)
  similarity = torch.max(similarity, dim=1).values.squeeze(-1)
  return similarity.cpu().numpy().tolist()


def image_reward(captions, image_paths):
  """Returns ImageReward rewards."""
  global ir_model
  if not ir_model: _init_reward_models('imagereward')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  rewards = []
  # scores = ir_model.score(captions, image_paths)
  # scores = scores if isinstance(scores, list) else [scores]
  # for i in range(len(image_paths)):
  #   per_image_scores = scores[len(captions)*i:len(captions)*(i+1)]
  #   rewards.append(per_image_scores[i])
  for caption, image_path in zip(captions, image_paths):
    rewards.append(ir_model.score(caption, [image_path]))
  return rewards


def pick_score(captions, image_paths):
  global pickscore_model
  if not pickscore_model: _init_reward_models('pickscore')

  captions = captions if isinstance(captions, list) else [captions]
  image_paths = image_paths if isinstance(image_paths, list) else [image_paths]

  # Preprocess.
  images = [Image.open(image_path) for image_path in image_paths]
  image_inputs = pickscore_processor(
      images=images, padding=True, truncation=True, max_length=77,
      return_tensors='pt').to(device)
  text_inputs = pickscore_processor(
      text=captions, padding=True, truncation=True, max_length=77,
      return_tensors='pt').to(device)

  with torch.no_grad():
    # Embeddings: [batch_size, 1024].
    image_embs = pickscore_model.get_image_features(**image_inputs)
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    text_embs = pickscore_model.get_text_features(**text_inputs)
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

    # Pick score.
    embs_dot = text_embs @ image_embs.T
    # Only take the scores for pairwise image and text features.
    embs_dot = torch.diagonal(embs_dot, 0)
    scores = pickscore_model.logit_scale.exp() * embs_dot / 100.0

  return scores.cpu().numpy().tolist()


base_prompt_dicts = {
  'compose_10': reward_fns.read_base_prompt_dict('compose_10'),
  'count_10': reward_fns.read_base_prompt_dict('count_10'),
  'open100_10': reward_fns.read_base_prompt_dict('open100_10'),
}

metadata_dicts = {
  'compose_10': reward_fns.read_metadata_dict('compose_10'),
  'count_10': reward_fns.read_metadata_dict('count_10'),
  'open100_10': reward_fns.read_metadata_dict('open100_10'),
}


def mean_ensemble_score(captions, image_paths, prompt_type):
  global ir_model, pickscore_model
  if not ir_model: _init_reward_models('imagereward')
  if not pickscore_model: _init_reward_models('pickscore')

  base_prompt_dict = base_prompt_dicts[prompt_type]
  metadata_dict = metadata_dicts[prompt_type]

  scores = []
  for caption, image_path in zip(captions, image_paths):
    base_prompts = base_prompt_dict[caption]
    base_prompts = set(base_prompts) - set([caption])
    base_prompts = [caption] + sorted(list(base_prompts))

    caption_image_paths = [image_path] * len(base_prompts)
    # To keep shapes consistent with that expected in reward_fns.py.
    ir_logits = [image_reward(base_prompts, caption_image_paths)]
    ps_logits = [pick_score(base_prompts, caption_image_paths)]
    scale = metadata_dict[caption]['mean_best']['scale']
    ensemble_scores, _ = reward_fns.mean_ensemble_score([ir_logits, ps_logits], scale)
    assert(len(ensemble_scores) == 1)
    scores.append(ensemble_scores[0])
  return scores


def uw_ensemble_score(captions, image_paths, prompt_type):
  global ir_model, pickscore_model
  if not ir_model: _init_reward_models('imagereward')
  if not pickscore_model: _init_reward_models('pickscore')

  base_prompt_dict = base_prompt_dicts[prompt_type]
  metadata_dict = metadata_dicts[prompt_type]

  scores = []
  for caption, image_path in zip(captions, image_paths):
    base_prompts = base_prompt_dict[caption]
    base_prompts = set(base_prompts) - set([caption])
    base_prompts = [caption] + sorted(list(base_prompts))

    caption_image_paths = [image_path] * len(base_prompts)
    # To keep shapes consistent with that expected in reward_fns.py.
    ir_logits = [image_reward(base_prompts, caption_image_paths)]
    ps_logits = [pick_score(base_prompts, caption_image_paths)]
    scale = metadata_dict[caption]['uw_best']['scale']
    lmbda = metadata_dict[caption]['uw_best']['lambda']
    ensemble_scores, ensemble_probs = reward_fns.mean_ensemble_score([ir_logits, ps_logits], scale)
    assert(len(ensemble_scores) == 1 and len(ensemble_probs) == 1)
    mean_score, probs = ensemble_scores[0], ensemble_probs[0]
    variance = np.mean((probs - mean_score) ** 2.0)
    uw_score = mean_score - lmbda * variance
    scores.append(uw_score)
  return scores


if __name__ == '__main__':
  image_fn = '/data/jongheon_jeong/dev/trl/analysis/deer_truck.png'

  scores = mean_ensemble_score(['a realistic photo of a deer and a truck'], [image_fn], 'compose_10')
  print(scores)
  scores = uw_ensemble_score(['a realistic photo of a deer and a truck'], [image_fn], 'compose_10')
  print(scores)
