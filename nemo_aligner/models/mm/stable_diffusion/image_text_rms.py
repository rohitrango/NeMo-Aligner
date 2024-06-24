# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize
import safetensors
import einops

from megatron.core import parallel_state
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import (
    CLIPTextTransformer,
    CLIPVisionTransformer,
    MegatronCLIPModel,
)
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal
from nemo.utils import logging
from nemo_aligner.data.mm.pickscore_dataset import build_train_valid_datasets
from open_clip import tokenize
from transformers import CLIPTokenizer
import math
from omegaconf import OmegaConf

BICUBIC = InterpolationMode.BICUBIC
BILINEAR = InterpolationMode.BILINEAR
PATCH_SIZE = 224
PATCH_SIZE2 = PATCH_SIZE//2

def get_idx(end, device):
    return torch.arange(start=0, end=end, dtype=torch.float32, device=device)

def coordinate_embedding(coordinates, dim, max_period=10000, repeat_only=False, cached_embedding=None):
    """
    Create sinusoidal embeddings from integer coordinates.

    Copied from nemo/collections/multimodal/modules/stable_diffusion/diffusionmodules/util.py

    Parameters:
        coordinates (Tensor): A 1-D tensor of N indices, one per batch element. These indices may be fractional and
                            represent the timesteps for which embeddings are to be created.
        dim (int): The dimension of the output embeddings. Each timestep will be represented as a vector of this dimension.
        max_period (float): Controls the minimum frequency of the embeddings. Higher values result in higher frequency
                            components in the embedding.

    Returns:
        Tensor: An [N x dim] tensor of positional embeddings, where each row corresponds to the embedding for a timestep.
    """
    if not repeat_only:
        if cached_embedding is not None:
            # using cached embedding and lookup in the cache
            embedding = cached_embedding[coordinates.to(dtype=torch.int), :]
        else:
            half = dim // 2
            idx = get_idx(half, coordinates.device)
            freqs = torch.exp(-math.log(max_period) / half * idx)
            args = coordinates[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = einops.repeat(coordinates, "b -> b d", d=dim)
    return embedding


class PickscoreRewardModel(MegatronModule):
    """CLIP-Based Model"""

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True):
        super(PickscoreRewardModel, self).__init__()
        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision, model_parallel_config, pre_process=self.pre_process, post_process=self.post_process,
        )
        self.text_encoder = CLIPTextTransformer(
            model_cfg.text,
            model_parallel_config,
            padded_vocab_size,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def get_reward(self, images, captions):
        text_features = self.text_encoder(captions)
        image_features = self.vision_encoder(images)
        rewards = (
            self.logit_scale.exp()
            * torch.matmul(F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1).t()).diag()
        )

        return rewards

class PickscoreMultiCropRewardModel(MegatronModule):
    """ CLIP-Based Model but supports multicrop since original CLIP model only supports 224x224 images """

    def __init__(self, model_cfg, model_parallel_config, padded_vocab_size, pre_process=True, post_process=True, seed=42):
        super(PickscoreMultiCropRewardModel, self).__init__()
        self.config = model_parallel_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.vision_encoder = CLIPVisionTransformer(
            model_cfg.vision, model_parallel_config, pre_process=self.pre_process, post_process=self.post_process,
        )
        # requires text tokens 
        self.text_encoder = CLIPTextTransformer(
            model_cfg.text,
            model_parallel_config,
            padded_vocab_size,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        attn_cfg = model_cfg.image_patch_attn
        sigma = attn_cfg.get('sigma', 0.02)
        num_layers = attn_cfg.get('num_layers', 2)

        # check for frozen
        if model_cfg.vision.freeze:
            logging.info("Freezing vision model.")
            self.vision_encoder.requires_grad_(False)
            self.vision_encoder.eval()
        
        if model_cfg.text.freeze:
            logging.info("Freezing text model.")
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        # get a bunch of sample resolutions for augmentation
        self.resolutions = model_cfg.get('sample_resolutions', [512, 768, 1024, 1280, 1536, 2048])
        logging.info(f"Sampling at resolutions {self.resolutions}.")

        # attention modules
        # TODO: Add some form of positional encoding to this layer 
        aggregator = model_cfg.get('aggregator', 'transformer')
        self.aggregator_name = aggregator
        if aggregator == 'transformer':
            logging.info("Using transformer aggregator.")
            self.aggregator = ParallelTransformer(
                model_parallel_config, 
                init_method=init_method_normal(sigma), 
                output_layer_init_method=scaled_init_method_normal(sigma, num_layers),
                num_layers=num_layers,
                ffn_hidden_size=model_cfg.output_dim * 4,
                hidden_size=model_cfg.output_dim * 2,
                num_attention_heads=attn_cfg.num_attention_heads,
                precision=model_cfg.precision,
            )
            self.final_out = nn.Linear(model_cfg.output_dim * 2, model_cfg.output_dim)
            # get linear embedder
            self.linear_embed = nn.Sequential(
                nn.Linear(model_cfg.output_dim, model_cfg.output_dim * 4),
                nn.SiLU(),
                nn.Linear(model_cfg.output_dim * 4, model_cfg.output_dim)
            )
            self.out_dim = model_cfg.output_dim


        elif aggregator == 'gap':
            # global average pooling
            logging.info("Using global average pooling.")
            self.aggregator = lambda x, *args, **kwargs: x    # aggregator is a no-op because we perform mean later
            if model_cfg.vision.freeze and model_cfg.text.freeze:
                assert False, "Need some module to be training when using Global Average Pooling aggregator"
        else:
            raise ValueError(f"Invalid aggregator {aggregator}.")

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        ## Load from pretrained clip model
        # TODO: Separate modules by names and then load them separately to keep track of missing/unexpected keys
        from_pretrained = model_cfg.get('from_pretrained')
        if from_pretrained is not None:
            logging.info(f"Loading pretrained model from {from_pretrained}")
            if from_pretrained.endswith('safetensors'):
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(from_pretrained)
            else:
                state_dict = torch.load(from_pretrained, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            # if it starts with model, strip it
            if all([x.startswith('model.') for x in state_dict.keys()]):
                state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
            # load state dict
            self.load_state_dict(state_dict, strict=False)

        self.rng = np.random.RandomState(seed)
        # this is to keep track of (top-left) grid locations to sample from for a given image size
        self.grid_locs = dict()
        

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def create_grid_locs(self, H, W):
        ''' given height and width, sample top-left locations to '''
        locs = []
        for l in range(0, W, PATCH_SIZE):
            l_correct = l - max(0, l + PATCH_SIZE - W)
            for t in range(0, H, PATCH_SIZE):
                t_correct = t - max(0, t + PATCH_SIZE - H)
                locs.append((l_correct, t_correct))
        locs = np.array(locs)
        return locs

    def get_image_crops(self, images):
        ''' images: bchw, we need to feed random or uniformly spaced 224 crops '''
        b, c, h, w = images.shape
        rng = self.rng
        patch_size = PATCH_SIZE
        # if doesnt exist, create, else use prev assigned
        if self.grid_locs.get((h, w)) is None:
            grid_locs = self.create_grid_locs(h, w)
            self.grid_locs[(h, w)] = grid_locs + 0
        else:
            grid_locs = self.grid_locs[(h, w)] + 0
        # randomly perturb them
        if self.training:
            grid_locs = grid_locs + rng.randint(-PATCH_SIZE//3, PATCH_SIZE//3, size=grid_locs.shape)
            grid_locs = np.maximum(0, grid_locs)
            grid_locs[:, 0] = np.minimum(w-PATCH_SIZE, grid_locs[:, 0])
            grid_locs[:, 1] = np.minimum(h-PATCH_SIZE, grid_locs[:, 0])
            # with a small probability, keep all these locs, or randomly dropout ~10% 
            if rng.rand() > 0.05:
                keep = np.where(rng.rand(grid_locs.shape[0]) < 0.9)[0]
                grid_locs = grid_locs[keep, :]
        
        # now sample the patches
        image_crops = []
        image_centers = []
        device = images.device
        for x, y in grid_locs:
            image_crops.append(images[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            image_centers.append(torch.FloatTensor([x + PATCH_SIZE2, y + PATCH_SIZE2]).unsqueeze(0).repeat(b, 1).to(device))  # [b, 2]
        return image_crops, image_centers

    def get_reward(self, images, captions):
        # images have to be resized, sample interpolation strategy and resolution
        # images   = list of tensors of size [3, H_i, W_i]
        # captions = list of text
        if self.training:
            res = int(self.rng.choice(self.resolutions))
            interp = ([BILINEAR, BICUBIC])[int(np.random.randint(2))]
        else:
            res = int(max([x.shape[-1] for x in images]))
            interp = BICUBIC

        # resample images 
        img_transform = Resize(res, interpolation=interp, antialias=True)
        images = [img_transform(x) for x in images]
        images = torch.stack(images, 0)
        # get crops 
        image_crops, image_centers = self.get_image_crops(images)  # bchw --> list [bch'w', bch'w' ....] , and list [B2, B2, ...] containing (cx, cy)
        num_crops = len(image_crops)
        ### select one crop to pass gradients through
        grad_idx = self.rng.randint(num_crops)
        encoded_features = []
        for i in range(num_crops):
            with torch.set_grad_enabled(i == grad_idx):
                encoded_features.append(self.vision_encoder(image_crops[i]))   # [B, D]
        # TODO: feed the image centers into pos encoding
        # concat and feed them to attention
        image_features = self.aggregate(torch.stack(encoded_features, dim=1), image_centers)
        text_features = self.text_encoder(captions)
        rewards = self.logit_scale.exp() * (F.normalize(image_features, dim=-1) * F.normalize(text_features, dim=-1)).sum(1)   # [B, ]
        return rewards
    
    def aggregate(self, encoded_features, image_centers):
        ''' 
        encoded_features: [B, S, D]
        image_centers: list of [B,2], [B,2] ... tensors

        use either GAP or transformer aggregation
        '''
        if self.aggregator_name == 'gap':
            image_features = self.aggregator(encoded_features).mean(1)
        elif self.aggregator_name == 'transformer':
            t_embed = []
            for center in image_centers:
                cx, cy = [center[:, i] for i in range(2)]
                embed = torch.cat([coordinate_embedding(coord, self.out_dim//2) for coord in [cx, cy]], dim=1)  # [B, out_dim]
                embed = self.linear_embed(embed)
                t_embed.append(embed)
            t_embed = torch.stack(t_embed, dim=1)  # [B, S, D]
            pos_features = torch.cat([encoded_features, t_embed], dim=2)
            pos_features = self.aggregator(pos_features, None)    # apply attention without any mask
            image_features = pos_features.mean(1)
            image_features = self.final_out(image_features)  # (2D -> D)
        else:
            raise ValueError("Invalid aggregation function.")
        return image_features

    def forward(self, images, captions):
        return self.get_reward(images, captions)

class MegatronCLIPRewardModel(MegatronCLIPModel):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        self.openai_dataset_mean = (0.48145466, 0.4578275, 0.40821073)
        self.openai_dataset_std = (0.26862954, 0.26130258, 0.27577711)
        self.transform_size = 224
        self.rescale_param = 0.00392156862745098
        self.differentiable_preprocess = self.diff_preprocess()

    def diff_preprocess(self):

        return Compose(
            [
                Resize(self.transform_size, interpolation=BICUBIC, antialias=True),
                CenterCrop(self.transform_size),
                self.rescale,
                Normalize(self.openai_dataset_mean, self.openai_dataset_std),
            ]
        )

    def rescale(self, image):
        return image * self.rescale_param

    def preprocess(self, images, captions):

        _, text_transform = get_preprocess_fns(self.cfg, tokenizer=self.tokenizer, is_train=False)

        images = (
            torch.stack([self.differentiable_preprocess(img.permute(2, 0, 1)) for img in images])
            .to(torch.cuda.current_device())
            .float()
        )

        captions_list = [text_transform(captions[i]) for i in range(images.shape[0])]

        captions = torch.stack(captions_list).to(torch.cuda.current_device())

        return images, captions

    def get_reward(self, images, captions):
        images, captions = self.preprocess(images, captions)
        return self.model.get_reward(images, captions)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = PickscoreRewardModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model
    
class MegatronCLIPMultiCropRewardModel(MegatronCLIPModel):
    def __init__(self, cfg, trainer):
        self.tokenizer = None
        super().__init__(cfg, trainer)
        self.openai_dataset_mean = (0.48145466, 0.4578275, 0.40821073)
        self.openai_dataset_std = (0.26862954, 0.26130258, 0.27577711)
        self.transform_size = 224
        # self.rescale_param = 0.00392156862745098   # we dont need this, as we assume the dataloader performs the preprocessing to openai mean/std 

    def preprocess(self, images, captions):
        # simply use openclip tokenizer which takes a list of strings and processes them into captions
        text_transform = tokenize
        images = [x.to(torch.cuda.current_device()).float() for x in images]
        captions = tokenize(captions).to(torch.cuda.current_device())
        # captions_list = [text_transform(captions[i]) for i in range(len(captions))]
        # captions = torch.stack(captions_list).to(torch.cuda.current_device())
        return images, captions

    def get_reward(self, images, captions):
        images, captions = self.preprocess(images, captions)
        return self.model.get_reward(images, captions)

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        model = PickscoreMultiCropRewardModel(
            model_cfg=self.cfg,
            model_parallel_config=self.model_parallel_config,
            padded_vocab_size=self.padded_vocab_size,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model
    
    def forward(self, images, captions):
        rewards = self.model(images, captions)
        return rewards
    
    def loss_func(self, output_tensor):
        ''' 
        rewards_0, rewards_1 - tensor of size [B,] each that computes rewards 
        labels - Bx2 tensor containing labels
        '''
        rewards_0, rewards_1, labels = output_tensor
        logits = torch.stack([rewards_0, rewards_1], 1)  # [B, 2]
        logsoftmax = F.log_softmax(logits, dim=1)
        labels_s = (labels + 1e-4)
        labels_s = labels_s / labels_s.sum(1, keepdim=True)
        # to balance out the non-zero term for 0.5 case
        entropy = (labels_s * torch.log(labels_s)).sum(1).mean().detach()
        # this is the KL div loss for categorical
        local_loss = -(labels * logsoftmax).sum(1).mean() + entropy
        # compute accuracy
        gt_exists = torch.where(labels.max(1).values > 0.5)[0]
        logits_masked = torch.argmax(logits, 1)[gt_exists]
        labels_masked = torch.argmax(labels, 1)[gt_exists]
        accuracy = torch.mean(1.0*(logits_masked == labels_masked))
        # compute reduced metrics over parallel group
        reduced_accuracy = average_losses_across_data_parallel_group([accuracy])
        reduced_loss = average_losses_across_data_parallel_group([local_loss])
        
        return local_loss, {"loss": reduced_loss, "accuracy": reduced_accuracy}

    # Override forward-backward function to train
    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch, _, _ = next(dataloader_iter)
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                # load images and caption
                img_0, img_1 = batch['img_0'], batch['img_1']
                img_0, img_1 = [x.to(torch.cuda.current_device()) for x in img_0], [x.to(torch.cuda.current_device()) for x in img_1]
                captions = batch['prompt'].to(img_0[0].device)
            else:
                raise NotImplementedError
                if parallel_state.is_pipeline_first_stage():
                    # first pipeline stage, prep images and captions
                    img_0, img_1 = batch['img_0'], batch['img_1']
                    captions = batch['prompt']
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    img_0, img_1, captions = None, None, None

            # output_tensor = model(images, captions)
            reward_0 = model(img_0, captions)
            reward_1 = model(img_1, captions)
            output_tensor = (reward_0, reward_1, batch['label'].to(torch.cuda.current_device()))
            return output_tensor, self.loss_func
        return fwd_output_and_loss_func

    def build_train_valid_test_datasets(self):
        logging.info('Building datasets for CLIP...')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        self._train_ds, self._validation_ds = build_train_valid_datasets(
            self.cfg, consumed_samples=self.compute_consumed_samples(0), tokenizer=self.tokenizer,
        )
        self._test_ds = None

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building datasets for CLIP.')
        return self._train_ds, self._validation_ds, self._test_ds


    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f'Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}'
            )
            self._train_dl = torch.utils.data.DataLoader(
                self._train_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True,
                collate_fn=self.dl_collate_fn,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds') and self._validation_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}'
            )
            self._validation_dl = torch.utils.data.DataLoader(
                self._validation_ds,
                batch_size=self._micro_batch_size,
                num_workers=cfg.num_workers,
                collate_fn=self.dl_collate_fn,
                pin_memory=True,
                drop_last=cfg.train.get("drop_last", True),
                persistent_workers=True if cfg.num_workers > 0 else False,
            )

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds') and self._test_ds is not None:
            consumed_samples = 0
            logging.info(
                f'Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}'
            )
            self._test_dl = torch.utils.data.DataLoader(
                self._test_ds, batch_size=self._micro_batch_size, num_workers=cfg.num_workers, pin_memory=True, 
                collate_fn=self.dl_collate_fn,
            )
    
    def dl_collate_fn(self, batch):
        ''' collate function for multi-crop reward model '''
        new_batch = {}
        keys = list(batch[0].keys())
        for k in keys:
            if k in ['img_0', 'img_1']:
                new_batch[k] = [datum[k] for datum in batch]
            elif k == 'prompt':
                # if prompts are strings, simply list them, else, stack them
                if isinstance(batch[0][k], str):
                    new_batch[k] = [datum[k] for datum in batch]
                else:
                    new_batch[k] = torch.stack([datum[k] for datum in batch], 0)
            else:
                new_batch[k] = torch.stack([datum[k] for datum in batch], 0)
        return new_batch


    def setup_optimizer_param_groups(self):
        """
            Megatron CLIP override

            Used to create param groups for the optimizer.
            As an example, this can be used to specify per-layer learning rates:

            optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                        ], lr=1e-2, momentum=0.9)

            See https://pytorch.org/docs/stable/optim.html for more information.
            By default, ModelPT will use self.parameters().
            Override this method to add custom param groups.
            In the config file, add 'optim_param_groups' to support different LRs
            for different components (unspecified params will use the default LR):

            model:
                optim_param_groups:
                    encoder:
                        lr: 1e-4
                        momentum: 0.8
                    decoder:
                        lr: 1e-3
                optim:
                    lr: 3e-3
                    momentum: 0.9
        """
        if not hasattr(self, "parameters"):
            self._optimizer_param_groups = None
            return

        vision_params, text_params, other_params = {'params': []}, {'params': []}, {'params': []}
        # check for which module they belong to
        for n, p in self.named_parameters():
            if n.startswith('vision_encoder.'):
                vision_params['params'].append(p)
            elif n.startswith('text_encoder.'):
                text_params['params'].append(p)
            else:
                other_params['params'].append(p)

        # for finetuning, we may have provided extra params
        if self.cfg.vision.get('extra_optim_params'):
            vision_params.update(OmegaConf.to_container(self.cfg.vision.extra_optim_params))
        if self.cfg.text.get('extra_optim_params'):
            text_params.update(OmegaConf.to_container(self.cfg.text.extra_optim_params))
        
        self._optimizer_param_groups = [vision_params, text_params, other_params]
        


def get_reward_model(cfg, mbs, gbs):
    ''' choose between the reward models '''
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        model_cfg.text.precision = cfg.trainer.precision
        if cfg.trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None
        model_cfg.global_batch_size = gbs
        model_cfg.micro_batch_size = mbs

    if cfg.get('multicrop', False):
        _, model = setup_trainer_and_model_for_inference(
            model_provider=MegatronCLIPMultiCropRewardModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
        )
    else:
        _, model = setup_trainer_and_model_for_inference(
            model_provider=MegatronCLIPRewardModel, cfg=cfg, model_cfg_modifier=model_cfg_modifier,
        )
    return model
