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
import itertools

import torch
from PIL import Image
from omegaconf import OmegaConf
import subprocess
from scipy import linalg
from pytorch_fid.fid_score import calculate_frechet_distance
from torch_fidelity.metric_kid import kid_features_to_metric


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

def get_pickscore_model(model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"):
    ''' initialize pickscore model from HF '''
    from transformers import AutoProcessor, AutoModel
    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
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
def clip_model_eval(args, model='pickscore'):
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

    if model == 'pickscore':
        model, processor = get_pickscore_model()
        print("Loaded pickscore model.")
    elif model == 'clip':
        model, processor = get_pickscore_model(model_pretrained_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        print("Loaded CLIP model.")
    else:
        print(f"Unknown model {model}.")
        return
    
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

def compute_fid_from_features(f1, f2, num_classes=40):
    ''' compute fid from features1 and features of size (M, D) and (N, D) '''
    assert (f1.shape[0] % num_classes) == 0 and (f2.shape[0] % num_classes) == 0 and num_classes >= 1
    m1, m2 = np.mean(f1, axis=0), np.mean(f2, axis=0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    # compute spectral distance
    eig1, eig2 = np.linalg.eigvals(s1), np.linalg.eigvals(s2)
    spec = np.linalg.norm(eig1 - eig2)
    logspec = np.linalg.norm(np.log10(eig1 + 1e-10) - np.log10(eig2 + 1e-10))        # log spectral distance
    # compute fid
    fid = calculate_frechet_distance(m1, s1, m2, s2)
    if num_classes == 1:
        return fid, fid, spec, logspec
    # compute classwise fid
    cfid = []
    m1pc, m2pc = f1.shape[0]//num_classes, f2.shape[0]//num_classes     # number of samples per class
    for i in tqdm(range(num_classes)):
        f1c, f2c = f1[i*m1pc:(i+1)*m1pc], f2[i*m2pc:(i+1)*m2pc]
        m1c, m2c = np.mean(f1c, axis=0), np.mean(f2c, axis=0)
        s1c, s2c = np.cov(f1c, rowvar=False), np.cov(f2c, rowvar=False)
        cfid.append(calculate_frechet_distance(m1c, s1c, m2c, s2c))
    return fid, np.mean(cfid), spec, logspec


def calc_cdist_part(features_1, features_2, batch_size=10000):
    dists = []
    for feat2_batch in features_2.split(batch_size):
        dists.append(torch.cdist(features_1, feat2_batch).cpu())
    return torch.cat(dists, dim=1)

# https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_prc.py#L27
def calculate_precision_recall_part(features_1, features_2, neighborhood=3, batch_size=10000):
    # Precision
    dist_nn_1 = []
    for feat_1_batch in features_1.split(batch_size):
        dist_nn_1.append(calc_cdist_part(feat_1_batch, features_1, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_1 = torch.cat(dist_nn_1)
    precision = []
    for feat_2_batch in features_2.split(batch_size):
        dist_2_1_batch = calc_cdist_part(feat_2_batch, features_1, batch_size)
        precision.append((dist_2_1_batch <= dist_nn_1).any(dim=1).float())
    precision = torch.cat(precision).mean().item()
    # Recall
    dist_nn_2 = []
    for feat_2_batch in features_2.split(batch_size):
        dist_nn_2.append(calc_cdist_part(feat_2_batch, features_2, batch_size).kthvalue(neighborhood + 1).values)
    dist_nn_2 = torch.cat(dist_nn_2)
    recall = []
    for feat_1_batch in features_1.split(batch_size):
        dist_1_2_batch = calc_cdist_part(feat_1_batch, features_2, batch_size)
        recall.append((dist_1_2_batch <= dist_nn_2).any(dim=1).float())
    recall = torch.cat(recall).mean().item()
    return precision, recall

@torch.no_grad()
def coverage_model_eval(args, base_model_filters=['base', 'kl0.0'], num_gpus=8):
    ''' 
    given a list of paths, find the base model, compute the discrepancy of all other models with respect to it
    '''
    # prepare paths
    paths = sorted(glob(args.paths))
    print(f"Found {len(paths)} paths.")
    is_base_path = [all([x in path for x in base_model_filters]) for path in paths]
    if np.array(is_base_path).astype(int).sum() < 1:
        raise ValueError("At least one path should satisfy the base criteria.")
    base_paths = []
    other_paths = []
    for i, basepathflag in enumerate(is_base_path):
        if basepathflag:
            print(f"{paths[i]} is a base path.")
            base_paths.append(paths[i])
        else:
            other_paths.append(paths[i])
    print("Generating FID stats...")
    # create commands to generate stats
    commands = [[] for _ in range(num_gpus)]
    for gpu, path in enumerate(paths):
        gpu = gpu % num_gpus
        imgpath = osp.join(path, "coverage")
        savepath = osp.join(path, "fid_stats.npz")
        if osp.exists(savepath):
            continue
        cmd = f"python -m pytorch_fid --device cuda:{gpu} --save-stats {imgpath} {savepath}"
        commands[gpu].append(cmd)
    # run all non-empty commands and wait
    commands = list(filter(lambda x: len(x) > 0, commands))
    commands = [" && ".join(commandlist) for commandlist in commands]
    for cmd in commands:
        print(cmd)
    processes = [subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in commands]
    # Wait for all processes to complete
    for proc in processes:
        proc.wait()  # Waits for each process to finish
    
    # finished saving fid stats, 
    for base_path in base_paths:
        data = np.load(osp.join(base_path, "fid_stats.npz"))
        base_features = data['mu']             # all samples, use modified pytorch_fid function to keep them
        base_features_torch = torch.FloatTensor(base_features)
        print(f"Base Path: {base_path}")
        for cmp_path in other_paths:
            print(f"Path: {cmp_path}")
            cmp_features = np.load(osp.join(cmp_path, "fid_stats.npz"))['mu']
            cmp_features_torch = torch.FloatTensor(cmp_features)
            # compute fid
            fid, classwise_fid, spectral_dist, log_spectral_dist = compute_fid_from_features(base_features, cmp_features, 1)
            prc, rcl = calculate_precision_recall_part(base_features_torch, cmp_features_torch , neighborhood=5,)
            kid = kid_features_to_metric(base_features_torch, cmp_features_torch, verbose=False)['kernel_inception_distance_mean']
            
            print(f"fid: {fid:.06f}")
            print(f"classwisefid: {classwise_fid:.06f}")
            print(f"spectral distance: {spectral_dist:.06f}")
            print(f"log spectral distance: {log_spectral_dist:.06f}")
            print(f"precision: {prc:.05f}")
            print(f"recall: {rcl:.05f}")
            print(f"kid: {kid:.05f}")
            print()

            # compute kid





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, required=True)
    parser.add_argument('--partiprompt_path', type=str, default="/opt/nemo-aligner/datasets/PartiPrompts.tsv")
    parser.add_argument('--version', type=str, default="v2.1")
    parser.add_argument('--pickscore_path', type=str, default="/opt/nemo-aligner/checkpoints/pickscore.nemo")
    
    # read args and call main
    args = parser.parse_args()
    # hps_eval(args)
    # clip_model_eval(args)
    # clip_model_eval(args, model='clip')
    coverage_model_eval(args)

