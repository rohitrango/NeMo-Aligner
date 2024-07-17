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

import torch
from PIL import Image

#from nemo_aligner.models.mm.stable_diffusion.image_text_rms import get_reward_model

def get_partiprompt_prompts(args):
    prompts = pd.read_csv(args.partiprompt_path, delimiter="\t")["Prompt"]
    prompts = list(prompts.iteritems())
    return prompts

def get_hps_model_preprocess(hps_version='v2.1'):
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

@torch.no_grad()
def hps_batch_process(images, prompts, batch_size=128, model=None, preprocess_val=None, tokenizer=None, hps_version='v2.1'):
    ''' given a list of images and prompts, batchify them and pass them into the model '''
    if model is None:
        model, preprocess_val, tokenizer = get_hps_model_preprocess(hps_version)
    results = []
    # batchify 
    imgprompts = list(zip(images, prompts))
    imgprompt_batch = [imgprompts[i:i+batch_size] for i in range(0, len(imgprompts), batch_size)]
    for batch in tqdm(imgprompt_batch):
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
    model, preprocessor, tokenizer = get_hps_model_preprocess(args.version)

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
def image_reward(args):
    ''' '''
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True)
    parser.add_argument('--partiprompt_path', type=str, default="/opt/nemo-aligner/datasets/PartiPrompts.tsv")
    parser.add_argument('--version', type=str, default="v2.1")
    
    # read args and call main
    args = parser.parse_args()
    hps_eval(args)
