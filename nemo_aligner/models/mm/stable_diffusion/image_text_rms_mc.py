import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize
import safetensors
import einops
from typing import List

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
import math
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

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
        # manually override postprocessing of vision and text encoders
        model_cfg = self.modify_model_cfg(model_cfg)
        self.vision_encoder.post_process = model_cfg.vision.post_process
        self.text_encoder.post_process = model_cfg.text.post_process

        # this is the configuration of the attention patches
        attn_cfg = model_cfg.image_patch_attn
        self.num_crops_grad = attn_cfg.get("num_crops_grad", 1)
        # save the model cfg for later
        self.model_cfg = model_cfg
        self.add_global_image_context = attn_cfg.get('add_global_image_context', False)
        if self.add_global_image_context:
            logging.info("Adding global context to the image encoding.")

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
        self.test_time_resolution = model_cfg.get('test_time_resolution', None)
        logging.info(f"Sampling at resolutions {self.resolutions}.")
        logging.info(f"Test time resolution set to {self.test_time_resolution}")

        # attention modules
        # TODO: Add some form of positional encoding to this layer 
        aggregator = model_cfg.get('aggregator', 'transformer')
        self.aggregator_name = aggregator
        self.set_aggregator(aggregator, attn_cfg, model_parallel_config)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        ## Load from pretrained clip model
        from_pretrained = model_cfg.get('from_pretrained')
        if from_pretrained is not None:
            self._load_state_dict_helper(from_pretrained)

        self.rng = np.random.RandomState(seed)
        # this is to keep track of (top-left) grid locations to sample from for a given image size
        self.grid_locs = dict()

    def modify_model_cfg(self, cfg):
        ''' modify config depending on aggregation modes etc '''
        agg = cfg.get('aggregator', 'transformer')
        # aggregation specific changes
        if agg == 'attn_patch':
            cfg.vision.post_process = False
            cfg.text.post_process = False
        if agg == 'attn_patch_nocrop':
            cfg.vision.post_process = False
            cfg.text.post_process = False
            # global aggregation is true
            cfg.image_patch_attn.add_global_image_context = True
        return cfg
    
    def set_aggregator(self, aggregator, attn_cfg, model_parallel_config):
        ''' set aggregator for the correct usage '''
        sigma = attn_cfg.get('sigma', 0.02)
        num_layers = attn_cfg.get('num_layers', 2)
        model_cfg = self.model_cfg
        # this flag helps us determine if we want to use image crops at all (yes by default)
        self.aggregate_crops = True
        # 
        if aggregator == 'transformer':
            # simple transformer applied on pooled CLS feature from image enc, and last token from text encoder
            logging.info("Using transformer aggregator.")
            self.aggregator = ParallelTransformer(
                model_parallel_config, 
                init_method=init_method_normal(sigma), 
                output_layer_init_method=scaled_init_method_normal(sigma, num_layers),
                num_layers=num_layers,
                ffn_hidden_size=model_cfg.output_dim * 8,
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
        
        elif aggregator == 'attn_patch_nocrop':
            ### in this mode, we will only use patch features from the full image (and not use any image crops)
            assert self.add_global_image_context, "Need to add global context in `attn_patch_nocrop` aggregation "
            self.aggregate_crops = False
            output_size = model_cfg.output_dim  # initialize output size to hidden size of vision encoder
            # get linear layers too
            self.vision_proj = nn.Linear(model_cfg.vision.hidden_size, output_size//2)
            self.text_proj   = nn.Linear(model_cfg.text.hidden_size, output_size//2)
            self.vision_token = nn.Parameter(torch.zeros(1, 1, output_size//2))
            self.text_token = nn.Parameter(torch.zeros(1, 1, output_size//2))
            self.transformer = ParallelTransformer(
                model_parallel_config, 
                init_method=init_method_normal(sigma), 
                output_layer_init_method=scaled_init_method_normal(sigma, num_layers),
                num_layers=num_layers,
                ffn_hidden_size=output_size*4,
                hidden_size=output_size,
                num_attention_heads=attn_cfg.num_attention_heads,
                precision=model_cfg.precision,
            )

        elif aggregator == 'attn_patch':
            ### failed model
            # we want to apply aggregation on image tokens, <delim>, text tokens
            logging.info("Using patch and token aggregation (crops only)")
            assert not self.add_global_image_context, "Cannot add global context in `attn_patch` aggregation "
            output_size = model_cfg.output_dim
            # get projections from vision and text 
            self.vision_proj = nn.Linear(model_cfg.vision.hidden_size, output_size)
            self.text_proj   = nn.Linear(model_cfg.text.hidden_size, output_size)
            self.delimiter_token = nn.Parameter(torch.zeros(1, 1, output_size))
            # this transformer takes the image and text contents and runs a transformer
            self.transformer = ParallelTransformer(
                model_parallel_config, 
                init_method=init_method_normal(sigma), 
                output_layer_init_method=scaled_init_method_normal(sigma, num_layers),
                num_layers=num_layers,
                ffn_hidden_size=output_size*4,
                hidden_size=output_size,
                num_attention_heads=attn_cfg.num_attention_heads,
                precision=model_cfg.precision,
            )
            # also another transformer that aggregates image features
            self.patch_transformer = ParallelTransformer(
                model_parallel_config, 
                init_method=init_method_normal(sigma), 
                output_layer_init_method=scaled_init_method_normal(sigma, num_layers),
                num_layers=num_layers,
                ffn_hidden_size=output_size*4,
                hidden_size=output_size,
                num_attention_heads=attn_cfg.num_attention_heads,
                precision=model_cfg.precision,
            )
            # this is for coordinate embedding 
            self.linear_embed = nn.Sequential(
                nn.Linear(output_size, output_size*4),
                nn.SiLU(),
                nn.Linear(output_size*4, output_size)
            )
            self.out_dim = model_cfg.output_dim
        else:
            raise ValueError(f"Invalid aggregator {aggregator}.")

    def _load_state_dict_helper(self, from_pretrained):
        ''' load state dict from pretrained model if specified '''
        logging.info(f"Loading pretrained model from {from_pretrained}")
        if from_pretrained.endswith('safetensors'):
            state_dict = load_safetensors(from_pretrained)
        else:
            state_dict = torch.load(from_pretrained, map_location='cpu')
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        # if it starts with model, strip it
        if all([x.startswith('model.') for x in state_dict.keys()]):
            state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
        # check loaded models
        loaded_keys = list(state_dict.keys())
        expected_keys = list(self.state_dict().keys())
        # missing and unexpected keys
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        # unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        if len(missing_keys) > 0 and (any([x.startswith("vision_encoder") for x in missing_keys]) or any([x.startswith("text_encoder") for x in missing_keys])):
            logging.warning(f"there are missing keys: {missing_keys}.")
        else:
            logging.info("there are no missing keys")
        # load state dict
        self.load_state_dict(state_dict, strict=False)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        # TODO (yuya): fix this
        pass

    def create_grid_locs(self, H, W):
        ''' given height and width, sample top-left locations to crop from '''
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
        device = images.device
        ## if the image is small enough, just return the center crop
        if h <= patch_size:
            return [images], [torch.FloatTensor([h//2, w//2]).unsqueeze(0).repeat(b, 1).to(device)]

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
            if rng.rand() > 0.25:
                keep = np.where(rng.rand(grid_locs.shape[0]) < 0.9)[0]
                grid_locs = grid_locs[keep, :]
        
        # now sample the patches
        image_crops = []
        image_centers = []
        for x, y in grid_locs:
            image_crops.append(images[:, :, y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            image_centers.append(torch.FloatTensor([x + PATCH_SIZE2, y + PATCH_SIZE2]).unsqueeze(0).repeat(b, 1).to(device))  # [b, 2]
        return image_crops, image_centers

    def get_reward(self, images, captions,):
        # images have to be resized, sample interpolation strategy and resolution
        # images   = list of tensors of size [3, H_i, W_i]
        # captions = tokenized text
        image_list = images
        image_features = 0

        # if this flag is set to true, we will perform some cropping and feature aggregation of the crops
        if self.aggregate_crops:
            if self.training:
                res = int(self.rng.choice(self.resolutions))
                interp = ([BILINEAR, BICUBIC])[int(np.random.randint(2))]
            else:
                # if there is a test-time resolution, use that
                if self.test_time_resolution is not None:
                    try:  # this can be a number or a string specifying how to choose target resolution
                        res = int(self.test_time_resolution)
                    except:
                        if res == 'min':
                            res = int(min([x.shape[-1] for x in images]))
                        elif res == 'max':
                            res = int(max([x.shape[-1] for x in images]))
                        else:
                            raise ValueError
                else:
                    res = int(max([x.shape[-1] for x in images]))
                interp = BICUBIC

            #### this portion is to get image crops
            # we will always use the vision encoder to output crop features
            # resample images to a target size
            img_transform = Resize(res, interpolation=interp, antialias=True)
            images = [img_transform(x) for x in images]
            images = torch.stack(images, 0)
            # get crops 
            image_crops, image_centers = self.get_image_crops(images)  # bchw --> list [bch'w', bch'w' ....] , and list [B2, B2, ...] containing (cx, cy)
            num_crops = len(image_crops)
            ### select one crop to pass gradients through
            grad_idx = self.rng.permutation(num_crops)[:self.num_crops_grad]
            encoded_features = []
            for i in range(num_crops):
                with torch.set_grad_enabled(i in grad_idx):
                    encoded_features.append(self.vision_encoder(image_crops[i]))   # [B, D] or [B, S, D]
            # concat and feed them to attention
            image_features = self.aggregate(encoded_features, image_centers, grad_idx)

        # add global context (i.e. entire image if provided)
        global_img_feature = 0
        if self.add_global_image_context:
            ctx_transform = Resize(PATCH_SIZE, interpolation=BICUBIC, antialias=True)
            resized_imgs = [ctx_transform(x) for x in image_list]
            resized_imgs = torch.stack(resized_imgs, 0)  # [b, c, h, w]
            global_img_feature = self._get_global_feature(resized_imgs)

        # capture text features and compute reward
        text_features = self.text_encoder(captions)  # [b, d] or [s, b, d]
        rewards = self._reward_fn(image_features, global_img_feature, text_features)
        return rewards
    
    def _get_global_feature(self, resized_imgs):
        # get separate (or same) features for the global images
        if self.aggregator_name in ['gap', 'transformer']:
            # in this case, all features are the pooled versions, so we simply compute the reward
            global_img_feature = self.vision_encoder(resized_imgs)
        elif self.aggregator_name == 'attn_patch':
            # shouldnt even be here
            return 0
        elif self.aggregator_name == 'attn_patch_nocrop':
            return self.vision_encoder(resized_imgs)
        else:
            raise ValueError(f"unsupported global image feature")
        return global_img_feature
    
    def _reward_fn(self, image_features, global_img_feature, text_features):
        ''' compute reward function from features '''
        if self.aggregator_name in ['gap', 'transformer']:
            # in this case, all features are the pooled versions, so we simply compute the reward
            rewards = self.logit_scale.exp() * (F.normalize(image_features + global_img_feature, dim=-1) * F.normalize(text_features, dim=-1)).sum(1)   # [B, ]

        elif self.aggregator_name == 'attn_patch':
            # image_features = [b, s, d]
            batch_size = image_features.shape[0]
            # reshape text features and pass it through encoder
            text_features = text_features.permute(1, 0, 2).contiguous()  # [b, s, d']
            text_features = text_features + self.text_proj(text_features)  # [b, s, d]

            all_features = torch.cat([image_features, self.delimiter_token.repeat(batch_size, 1, 1), text_features], dim=1)  # [b, S, d]
            # run this through the transformer (after (b, s, d) -> (s, b, d))
            out_features = self.transformer(all_features.permute(1, 0, 2).contiguous(), None)  # [s, b, d]
            final_img_feature, final_text_feature = out_features[0], out_features[-1]
            rewards = self.logit_scale.exp() * (F.normalize(final_img_feature, dim=1) * F.normalize(final_text_feature, dim=-1)).sum(1)
        
        elif self.aggregator_name == 'attn_patch_nocrop':
            # global_img_feature = [B, S, D]
            # global_text_feature = [S, B, D]
            B = global_img_feature.shape[0]
            text_features = text_features.permute(1, 0, 2).contiguous() # [b, s, d]
            img_f = torch.cat([self.vision_proj(global_img_feature), self.vision_token.repeat(B, global_img_feature.shape[1], 1)], dim=2)
            txt_f = torch.cat([self.text_proj(text_features), self.text_token.repeat(B, text_features.shape[1], 1)], dim=2)
            full_f = torch.cat([img_f, txt_f], dim=1).permute(1, 0, 2).contiguous()
            full_f = self.transformer(full_f, None)
            final_img_feature, final_text_feature = full_f[0], full_f[-1]
            rewards = self.logit_scale.exp() * (F.normalize(final_img_feature, dim=1) * F.normalize(final_text_feature, dim=-1)).sum(1)

        else:
            raise ValueError(f"unsupported aggregation")
        return rewards
    
    def aggregate(self, encoded_features: List[torch.Tensor], image_centers: List[torch.Tensor], grad_idx: List[int]):
        ''' 
        encoded_features: stacked [B, S, D] or list of tensors
        image_centers: list of [B,2], [B,2] ... tensors

        use either GAP or transformer aggregation
        '''
        if self.aggregator_name == 'gap':
            image_features = torch.stack(encoded_features, dim=1)         # [b, s, d]
            image_features = self.aggregator(image_features).mean(1)  # [b, d]
            
        elif self.aggregator_name == 'transformer':
            encoded_features = torch.stack(encoded_features, dim=1)
            t_embed = self.get_pos_embed(image_centers)
            # get total features
            pos_features = torch.cat([encoded_features, t_embed], dim=2)  # [b, s, d]
            # aggregator takes input [s, b, d] so use permute
            pos_features = self.aggregator(pos_features.permute(1, 0, 2).contiguous(), None).permute(1, 0, 2).contiguous()    # apply attention without any mask
            image_features = pos_features.mean(1)
            image_features = self.final_out(image_features)  # (2D -> D)

        elif self.aggregator_name == 'attn_patch':
            # image_features is a list of sequence of tokens (b, s, d)
            num_crops = len(encoded_features)
            image_features = []
            t_embed = self.get_pos_embed(image_centers)  # [B, ncr, D]
            for i in range(num_crops):
                with torch.set_grad_enabled(i in grad_idx):
                    # compute vision projection and train tiny-transformer on top
                    vision_proj = self.vision_proj(encoded_features[i])  # [b, s, d]
                    vision_proj = vision_proj + t_embed[:, i:i+1, :]
                    vision_proj = self.patch_transformer(vision_proj.permute(1, 0, 2).contiguous(), None).permute(1, 0, 2).contiguous()
                    image_features.append(vision_proj)
                    #vision_proj = self.patch_transformer(vision_proj.permute(1, 0, 2).contiguous(), None).mean(0)  # [b, d]
                    # image_features.append(t_embed[:, i, :] + vision_proj)   # each element is [B, D]
            # assign it to stacked values
            image_features = torch.stack(image_features, dim=1).mean(1)  # [B, S, D]
        else:
            raise ValueError("Invalid aggregation function.")
        return image_features

    def forward(self, images, captions):
        return self.get_reward(images, captions)
    
    def get_pos_embed(self, image_centers):
        ''' get embeddings from image centers which is a list of tensors each of size (b, 2) '''
        t_embed = []
        for center in image_centers:
            cx, cy = [center[:, i] for i in range(2)]
            embed = torch.cat([coordinate_embedding(coord, self.out_dim//2) for coord in [cx, cy]], dim=1)  # [B, out_dim]
            embed = self.linear_embed(embed)
            t_embed.append(embed)
        t_embed = torch.stack(t_embed, dim=1)  # [B, S, D]            
        return t_embed


class MegatronCLIPMultiCropRewardModel(MegatronCLIPModel):
    def __init__(self, cfg, trainer):
        super().__init__(cfg, trainer)
        self.openai_dataset_mean = (0.48145466, 0.4578275, 0.40821073)
        self.openai_dataset_std = (0.26862954, 0.26130258, 0.27577711)
        self.transform_size = 224
        self.rescale_param = 0.00392156862745098   # we dont need this, as we assume the dataloader performs the preprocessing to openai mean/std 
        self.differentiable_preprocess = self.diff_preprocess()
        _, self.text_transform = get_preprocess_fns(self.cfg, tokenizer=self.tokenizer, is_train=False) 

    def rescale(self, image):
        return image * self.rescale_param

    def diff_preprocess(self):
        return Compose(
            [
                self.rescale,
                Normalize(self.openai_dataset_mean, self.openai_dataset_std),
            ]
        )

    def preprocess(self, images, captions):
        text_transform = self.text_transform
        images = [self.differentiable_preprocess(img.permute(2, 0, 1)).to(torch.cuda.current_device()).float() for img in images]
        captions_list = [text_transform(captions[i]) for i in range(len(images))]
        captions = torch.stack(captions_list).to(torch.cuda.current_device())
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
        rewards = self.get_reward(images, captions)
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
                captions = batch['prompt']
            else:
                raise NotImplementedError
                if parallel_state.is_pipeline_first_stage():
                    # first pipeline stage, prep images and captions
                    img_0, img_1 = batch['img_0'], batch['img_1']
                    captions = batch['prompt']
                else:
                    # Intermediate / Last pipeline stage doesn't need any inputs
                    img_0, img_1, captions = None, None, None

            reward_0 = self(img_0, captions)
            reward_1 = self(img_1, captions)
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
            if n.startswith('vision_encoder.'): # or n.startswith('logit_scale'):
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
        
