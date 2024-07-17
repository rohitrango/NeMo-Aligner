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

import torch
import torch.distributed
import torch.multiprocessing as mp
from megatron.core import parallel_state
from megatron.core.utils import divide
from omegaconf.omegaconf import OmegaConf, open_dict
from copy import deepcopy
import os
from functools import partial
from torch import nn
import numpy as np
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name
from PIL import Image
import pandas as pd
from os import path as osp
from tqdm import tqdm
import hpsv2

from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronStableDiffusionTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo_aligner.algorithms.supervised import SupervisedTrainer
from nemo_aligner.data.mm import text_webdataset
from nemo_aligner.data.nlp.builders import build_dataloader
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import get_reward_model
from nemo_aligner.models.mm.stable_diffusion.megatron_sdxl_draftp_model import MegatronSDXLDRaFTPModel
from nemo_aligner.models.mm.stable_diffusion.megatron_sd_draftp_model import MegatronSDDRaFTPModel
from nemo_aligner.utils.distributed import Timer
from nemo_aligner.utils.train_script_utils import (
    CustomLoggerWrapper,
    add_custom_checkpoint_callback,
    extract_optimizer_scheduler_from_ptl_model,
    init_distributed,
    init_peft,
    init_using_ptl,
    retrieve_custom_trainer_state_dict,
    temp_pop_from_config,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    LatentDiffusion,
    MegatronLatentDiffusion,
)
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine, DiffusionEngine
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPFSDPStrategy
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.openaimodel import UNetModel, ResBlock, SpatialTransformer, TimestepEmbedSequential
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder import AutoencoderKL, AutoencoderKLInferenceWrapper
from nemo.collections.multimodal.modules.stable_diffusion.diffusionmodules.model import Encoder, Decoder, ResnetBlock, AttnBlock
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import MegatronCLIPRewardModel
from nemo.collections.multimodal.modules.stable_diffusion.encoders.modules import FrozenOpenCLIPEmbedder, FrozenOpenCLIPEmbedder2, FrozenCLIPEmbedder
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import ParallelLinearAdapter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)

mp.set_start_method("spawn", force=True)

def get_weight_fn(wt_type: str):
    ''' get function to do weighing '''
    if wt_type == 'base':
        wt_draft = lambda sigma, sigma_next, i, total: 0
    elif wt_type == 'linear':
        wt_draft = lambda sigma, sigma_next, i, total: i*1.0/total
    elif wt_type == 'draft':
        wt_draft = lambda sigma, sigma_next, i, total: 1
    elif wt_type.startswith('power'):  # its of the form power_{power}
        pow = float(wt_type.split("_")[1])
        wt_draft = lambda sigma, sigma_next, i, total: (i*1.0/total)**pow
    elif wt_type.startswith("step"):   # use a step function (step_{p})
        frac = float(wt_type.split("_")[1])
        wt_draft = lambda sigma, sigma_next, i, total: float((i*1.0/total) >= frac)
    else:
        raise ValueError(f"invalid weighing type: {wt_type}")
    return wt_draft

def resolve_and_create_trainer(cfg, pop_trainer_key):
    """resolve the cfg, remove the key before constructing the PTL trainer
        and then restore it after
    """
    OmegaConf.resolve(cfg)
    with temp_pop_from_config(cfg.trainer, pop_trainer_key):
        return MegatronStableDiffusionTrainerBuilder(cfg).create_trainer()
    
def get_latents(batch_size, ptl_model, gen):
    ''' get latent vectors '''
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        latents = torch.randn(
            [
                batch_size,
                ptl_model.in_channels,
                ptl_model.height // ptl_model.downsampling_factor,
                ptl_model.width // ptl_model.downsampling_factor,
            ],
            generator=gen,
        ).to(torch.cuda.current_device())
    return latents

@torch.no_grad()
@hydra_runner(config_path="conf", config_name="draftp_sd")
def main(cfg) -> None:

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # set cuda device for each process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)

    # turn off wandb logging
    cfg.exp_manager.create_wandb_logger = False
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg.model.data.train.dataset_path = [cfg.model.data.webdataset.local_root_path for _ in range(cfg.trainer.devices * cfg.trainer.num_nodes)]
    cfg.model.data.validation.dataset_path = [
        cfg.model.data.webdataset.local_root_path for _ in range(cfg.trainer.devices * cfg.trainer.num_nodes)
    ]

    trainer = resolve_and_create_trainer(cfg, "draftp_sd")
    save_root_dir = exp_manager(trainer, cfg.exp_manager)
    logger = CustomLoggerWrapper(trainer.loggers)
    # Instatiating the model here
    ptl_model = MegatronSDDRaFTPModel(cfg.model, trainer).to(torch.cuda.current_device())
    init_peft(ptl_model, cfg.model)   # init peft 

    trainer_restore_path = trainer.ckpt_path

    if trainer_restore_path is not None:
        custom_trainer_state_dict = retrieve_custom_trainer_state_dict(trainer)
        consumed_samples = custom_trainer_state_dict["consumed_samples"]
    else:
        custom_trainer_state_dict = None
        consumed_samples = 0

    init_distributed(trainer, ptl_model, cfg.model.get("transformer_engine", False))

    # use the validation ds if needed
    train_ds, validation_ds = text_webdataset.build_train_valid_datasets(
        cfg.model.data, consumed_samples=consumed_samples
    )
    validation_ds = [d["captions"] for d in list(validation_ds)]

    val_dataloader = build_dataloader(
        cfg,
        dataset=validation_ds,
        consumed_samples=consumed_samples,
        mbs=cfg.model.micro_batch_size,
        gbs=cfg.model.global_batch_size,
        load_gbs=True,
    )
    init_using_ptl(trainer, ptl_model, val_dataloader, validation_ds)
    
    if cfg.model.get('activation_checkpointing', False):
        # call activation checkpointing here
        logging.info("Applying activation checkpointing on UNet and Decoder.")
        non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        def checkpoint_check_fn(module):
            return isinstance(module, (Decoder, UNetModel, MegatronCLIPRewardModel))
        apply_activation_checkpointing(ptl_model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=checkpoint_check_fn)

    optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(ptl_model)
    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)
    logger.log_hyperparams(OmegaConf.to_container(cfg))
    reward_model = get_reward_model(cfg.rm, mbs=cfg.model.micro_batch_size, gbs=cfg.model.global_batch_size)
    ptl_model.reward_model = reward_model

    torch.distributed.barrier()

    ckpt_callback = add_custom_checkpoint_callback(trainer, ptl_model)
    timer = Timer(cfg.exp_manager.get("max_time_per_run", "0:03:55:00"))   # save a model just before 4 hours

    draft_p_trainer = SupervisedTrainer(
        cfg=cfg.trainer.draftp_sd,
        model=ptl_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=val_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=[],
        logger=logger,
        ckpt_callback=ckpt_callback,
        run_timer=timer,
        run_init_validation=True,
    )

    if custom_trainer_state_dict is not None:
        draft_p_trainer.load_state_dict(custom_trainer_state_dict)

    torch.cuda.empty_cache()

    ### save_root_dir is already extracted from exp_manager
    ### Run PartiPrompts
    batch_size = 8
    partiprompts_data_path = "/opt/nemo-aligner/datasets/PartiPrompts.tsv"
    partiprompts = pd.read_csv(partiprompts_data_path, delimiter="\t")['Prompt']
    partiprompts = partiprompts[local_rank::world_size]
    partiprompts = list(partiprompts.iteritems())
    # batchify
    partiprompts_batch = [partiprompts[i:i+batch_size] for i in range(0, len(partiprompts), batch_size)]

    # save path for the images

    # run model for all weighing types (default is just draft)
    weighing_types = cfg.get("weight_type", "draft").split(",")
    for wt in weighing_types:
        wt_fn = get_weight_fn(wt)

        # reset generator for each type of weighing to compare effect of seed
        gen = torch.Generator(device='cpu')
        gen.manual_seed((1243 + 1247837 * local_rank)%(int(2**32 - 1)))

        # create partiprompts save_path
        pp_save_path = osp.join(save_root_dir, "saved_images", wt, "partiprompts")
        if osp.exists(pp_save_path):
            logging.info("Partiprompts path already exists, skipping. If you want to re-run, delete existing folder.")
            partiprompts_batch = []
        # everyone has to wait for the check (so that one process doesnt create the dir first) 
        torch.distributed.barrier()
        if local_rank == 0:
            os.makedirs(pp_save_path, exist_ok=True)
        torch.distributed.barrier()
        # let rank 0 make the dir so that others dont write to another dir

        logging.info("Generating partiprompt images...")
        if local_rank == 0:
            partiprompts_batch = tqdm(partiprompts_batch)
        # save partiprompt images
        for batch in partiprompts_batch:
            # divide into indices and prompts
            indices = [x[0] for x in batch]
            prompts = [x[1] for x in batch]
            # generate image
            latents = get_latents(len(prompts), ptl_model, gen)
            # TODO(@rohitrango): make this also return kl values
            images, = ptl_model.annealed_guidance(prompts, latents, weighing_fn=wt_fn, return_kl=True)
            images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)  # outputs are already scaled from [0, 255]
            for index, image in zip(indices, images):
                Image.fromarray(image).save(osp.join(pp_save_path, f"{index:05d}.png"))
        logging.info("Saved partiprompts images.")
        
        # Save HPSv2 images
        hpsv2_save_path = osp.join(save_root_dir, "saved_images", wt, "hpsv2")
        if local_rank == 0:
            os.makedirs(hpsv2_save_path, exist_ok=True)
        # get hpsv2 prompts, and sample them by world size and local rank
        all_prompts = hpsv2.benchmark_prompts('all') 
        for style, prompts_list in all_prompts.items():
            # get list of prompts
            prompts_list = list(enumerate(prompts_list))
            prompts_list = prompts_list[local_rank::world_size]
            prompts_batch = [prompts_list[i:i+batch_size] for i in range(0, len(prompts_list), batch_size)]
            if local_rank == 0:
                prompts_batch = tqdm(prompts_batch)
            # check if this subdir exists
            if osp.exists(osp.join(hpsv2_save_path, style)):
                logging.info(f"HPSv2 {style} path exists, skipping...")
                prompts_batch = []
            torch.distributed.barrier()
            if local_rank == 0:
                os.makedirs(osp.join(hpsv2_save_path, style), exist_ok=True)
            torch.distributed.barrier()
            # enumerate over batches
            for batch in prompts_batch:
                indices, prompts = [[x[i] for x in batch] for i in range(2)]
                # run diffusion
                latents = get_latents(len(prompts), ptl_model, gen)
                images = ptl_model.annealed_guidance(prompts, latents, weighing_fn=wt_fn)
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)  # outputs are already scaled from [0, 255]
                for index, image in zip(indices, images):
                    Image.fromarray(image).save(osp.join(hpsv2_save_path, style, f"{index:05d}.jpg"))
            logging.info(f"Saved HPSv2 images with style: {style}.")
    


if __name__ == "__main__":
    main()
