''' given an image directory, check partiprompts and HPSv2 folders and return all rewards '''
import torch
import argparse
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from os import path as osp
import warnings
import argparse
import os
import requests
import json

import torch
from PIL import Image
from omegaconf import OmegaConf


def get_partiprompt_prompts(args):
    prompts = pd.read_csv(args.partiprompt_path, delimiter="\t")["Prompt"]
    prompts = list(prompts.iteritems())
    return prompts

def get_hps_model_preprocessor(hps_version='v2.1'):
    ''' initialize the HPS model for one-time processing '''
    from hpsv2.img_score import initialize_model, model_dict
    from hpsv2.utils import root_path, hps_version_map
    from hpsv2.src.open_clip import get_tokenizer
    import huggingface_hub

    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # check if the checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    # load checkpoint
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

    checkpoint = torch.load(cp, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to('cuda')
    model.eval()
    return model, preprocess_val, tokenizer

def get_pickscore_model():
    ''' initialize pickscore model from HF '''
    from transformers import AutoProcessor, AutoModel
    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
    # get model and processor
    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    return model, processor

### Scripts to process batches of images

@torch.no_grad()
def pickscore_batch_process(images, prompts, model, processor, batch_size=128, pbar=True):
    ''' given list of images/prompts, pass them into model '''

    def norm_fn(tensor, **kwargs):
        norm = torch.norm(tensor, **kwargs)
        return tensor / norm

    scores = []
    # batchify 
    imgprompts = list(zip(images, prompts))
    imgprompt_batch = [imgprompts[i:i+batch_size] for i in range(0, len(imgprompts), batch_size)]
    pbar_wrapper = tqdm if pbar else lambda x: x
    for batch in pbar_wrapper(imgprompt_batch):
        images, prompts = [[x[i] for x in batch] for i in range(2)]
        images = [Image.open(x) for x in images]
        images = processor(images=images, return_tensors='pt').to('cuda')
        prompts = processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
        # get embeds
        img_emb = norm_fn(model.get_image_features(**images), dim=-1, keepdim=True)
        text_emb = norm_fn(model.get_text_features(**prompts), dim=-1, keepdim=True)
        # score
        score = model.logit_scale.exp() * (img_emb * text_emb).sum(1)  # [B, ]
        scores.append(score.cpu().numpy())
    scores = np.concatenate(scores, 0)
    return scores

@torch.no_grad()
def hps_batch_process(images, prompts, batch_size=128, model=None, preprocess_val=None, tokenizer=None, hps_version='v2.1', pbar=True):
    ''' given a list of images and prompts, batchify them and pass them into the model '''
    if model is None:
        model, preprocess_val, tokenizer = get_hps_model_preprocessor(hps_version)
    results = []
    # batchify 
    imgprompts = list(zip(images, prompts))
    imgprompt_batch = [imgprompts[i:i+batch_size] for i in range(0, len(imgprompts), batch_size)]
    pbar_wrapper = tqdm if pbar else lambda x: x
    for batch in pbar_wrapper(imgprompt_batch):
        images, prompts = [[x[i] for x in batch] for i in range(2)]
        images = [preprocess_val(Image.open(img)).unsqueeze(0).to(device='cuda', non_blocking=True) for img in images]
        images = torch.cat(images, dim=0)  # bchw
        # process tokens
        text = tokenizer(prompts).to(device='cuda', non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            #logits_per_image = image_features @ text_features.T
            hps_score = (image_features * text_features).sum(1).cpu().numpy() * 100
            results.append(hps_score)
    return np.concatenate(results, 0)

@torch.no_grad()
def hps_eval(args):
    ''' Perform HPSv2 eval on all the generated images '''
    import hpsv2

    paths = sorted(glob(args.paths))
    print(f"Found {len(paths)} paths.")
    # get partiprompt prompts
    indexed_pp_prompts = get_partiprompt_prompts(args)
    # get hps model
    model, preprocessor, tokenizer = get_hps_model_preprocessor(args.version)

    for path in paths:
        print(f"Path: {path}")
        pp_images = sorted(glob(osp.join(path, 'partiprompts', "*.png")))
        hpsv2_path = osp.join(path, 'hpsv2')
        # get HPSv2 result 
        hpsv2_result = hpsv2.evaluate(hpsv2_path, args.version)

        # compute PartiPrompt results
        if len(indexed_pp_prompts) == len(pp_images):
            prompts = [x[1] for x in indexed_pp_prompts]
            scores = hps_batch_process(pp_images, prompts, model=model, preprocess_val=preprocessor, tokenizer=tokenizer)
            print(f"PartiPrompts: {np.mean(scores):.04f} +- {np.std(scores):.04f}")
        else:
            print("Mismatch in number of prompts and images", len(indexed_pp_prompts), len(pp_images))
        # compute results on custom eval

@torch.no_grad()
def pickscore_eval(args):
    ''' use pickscore model '''
    import hpsv2
    from hpsv2 import root_path as meta_dir
    from hpsv2.utils import download_benchmark_prompts
    # download benchmark prompts in the beginning
    download_benchmark_prompts()
    hps_prompt_root = glob(os.path.join(meta_dir, 'datasets/benchmark/*.json'))
    hps_prompt_dict = dict()
    for file in hps_prompt_root:
        name = file.split("/")[-1].split(".")[0]
        with open(file, 'r') as fi:
            hps_prompt_dict[name] = json.load(fi)
    print("Loaded promptlist for HPSv2.")

    model, processor = get_pickscore_model()
    print("Loaded pickscore model.")
    
    # prepare paths
    paths = sorted(glob(args.paths))
    print(f"Found {len(paths)} paths.")
    # get partiprompt prompts
    indexed_pp_prompts = get_partiprompt_prompts(args) 
    pp_prompts = [x[1] for x in indexed_pp_prompts]

    # this is model paths
    for path in paths:
        print(path)

        ### compute PartiPrompt results
        pp_images = sorted(glob(osp.join(path, 'partiprompts', "*.png")))
        if len(indexed_pp_prompts) == len(pp_images):
            scores = pickscore_batch_process(pp_images, pp_prompts, model, processor,)
            print(f"PartiPrompts: {np.mean(scores):.04f} +- {np.std(scores):.04f}")
        else:
            print("Mismatch in number of prompts and images", len(indexed_pp_prompts), len(pp_images))

        ### get HPSv2 images
        hpsv2_paths = sorted(glob(osp.join(path, 'hpsv2', '*')))
        for hpsv2path in hpsv2_paths:
            stylename = hpsv2path.split("/")[-1] 
            if stylename == '':
                stylename = hpsv2path.split("/")[-2] 
            # given stylename, get images and prompts
            hps_prompts = hps_prompt_dict[stylename]
            hps_images = sorted(glob(osp.join(hpsv2path, "*.jpg")))
            if len(hps_images) == len(hps_prompts):
                scores = pickscore_batch_process(hps_images, hps_prompts, model, processor, pbar=False)
                spacestr = max([len(x) for x in hps_prompt_dict.keys()]) - len(stylename)
                spacestr = " "*spacestr
                print(f"HPSv2 ({stylename}){spacestr} : {np.mean(scores):.04f} +- {np.std(scores):.04f}")
            else:
                print("Mismatch in number of prompts and images", len(hps_prompts), len(hps_images))
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True)
    parser.add_argument('--partiprompt_path', type=str, default="/opt/nemo-aligner/datasets/PartiPrompts.tsv")
    parser.add_argument('--version', type=str, default="v2.1")
    parser.add_argument('--pickscore_path', type=str, default="/opt/nemo-aligner/checkpoints/pickscore.nemo")
    
    # read args and call main
    args = parser.parse_args()
    # hps_eval(args)
    pickscore_eval(args)
