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

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import io
from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
import torch
from torch.utils.data import Dataset, default_collate
from nemo.utils import logging
from torchvision.transforms import (
        Compose,
        Normalize,
        ToTensor,
    )
# load files
from datasets import Dataset as Dataset_hf, concatenate_datasets
from glob import glob

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def build_train_valid_datasets(
    model_cfg, consumed_samples, tokenizer=None, seed=None,
):
    train_data = PickScoreDataset(
        model_cfg,
        tokenizer=tokenizer,
        consumed_samples=consumed_samples,
        split='train',
        seed=seed,
    )

    val_data = PickScoreDataset(
        model_cfg,
        tokenizer=tokenizer,
        consumed_samples=consumed_samples,
        split='val',
        seed=seed,
    )

    # TODO: Add test data
    return train_data, val_data

    
class PickScoreDataset(Dataset): 
    def __init__(
        self,
        model_cfg,
        tokenizer,
        stop_idx = None,
        consumed_samples = 0,
        path = None,
        seed: int = 42,
        split: str = "train",
    ):
        super().__init__()
        self.model_cfg = model_cfg
        # check path
        self.tokenizer = tokenizer
        assert split in ("train", "val", "test")
        # self.split = split
        self.split_path = {"train": "train", "val": "validation", "test": "test"}[split]
        self.path = path or model_cfg.data.get('data_path')
        # lazy load all datasets
        from os import path as osp
        datasets = sorted(list(glob(osp.join(self.path, self.split_path, "*.arrow"))))
        datasets = [Dataset_hf.from_file(x) for x in datasets]
        datasets = concatenate_datasets(datasets)
        self.df = datasets
        num_rows = datasets.num_rows
        print(f"*********** Loading {split} dataset containing {num_rows} entries ***********")
        # shuffle the indices if given
        self.shuffled_indices = np.arange(num_rows).astype(np.int32)
        if seed is not None:
            np_rng = np.random.RandomState(seed=seed)
            np_rng.shuffle(self.shuffled_indices)
        
        # Get image and text transforms
        image_transform, self.text_transform = get_preprocess_fns(self.model_cfg, tokenizer = self.tokenizer, is_train=split == 'train')
        # create custom image transform 
        # this is the default behavior, simply convert to tensor and normalize
        if model_cfg.data.get('no_crop_images', True):
            logging.info("Creating custom image transform that does not resize images to 224x224.")
            self.image_transform = Compose([
                ToTensor(),
                Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)]
            )
        else:
            logging.info("Using default CLIP image transform.")
            self.image_transform = image_transform


    def __len__(self) -> int:
           return len(self.shuffled_indices)
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        true_idx = int(self.shuffled_indices[i])
        df = self.df[true_idx]
        img_0 = Image.open(io.BytesIO(df['jpg_0'])).convert("RGB")
        img_1 = Image.open(io.BytesIO(df['jpg_1'])).convert("RGB")
        label = torch.FloatTensor([df['label_0'], df['label_1']])     # preference label
        text = df['caption']

        img_0, img_1 = self.image_transform(img_0), self.image_transform(img_1)
        text = self.text_transform(text)

        output = {
            'img_0': img_0,
            'img_1': img_1,
            'label': label,
            'prompt': text,
            'time_step': torch.tensor([0.])
        }
        return output


if __name__ == '__main__':
    from open_clip import tokenizer
    from omegaconf import OmegaConf
    cfg = {
        'no_crop_images': True,
        'vision': {
            'img_w': 224,
            'img_h': 224,
            'img_mean': OPENAI_DATASET_MEAN,
            'img_std': OPENAI_DATASET_STD
        },
        'text': {
            'max_position_embeddings': 77,
        },
        'data_path': '/opt/nemo-aligner/datasets/pickscore', 
    }
    cfg = OmegaConf.create(cfg)
    dataset = PickScoreDataset(cfg, tokenizer=None, split='train')
    print(len(dataset))
    for i in range(10):
        batch = dataset[i]
        for k, v in batch.items():
            if k == 'prompt':
                print(k, v)
            else:
                print(k, v.shape)
        print()
