from glob import glob
import numpy as np
import re
from collections import defaultdict
from os import path as osp
import json
import torch

files = [
    'sdxl_clip_eval.txt',
    'sdxl_pickscore_eval.txt',
    'sdxl_hpsv2.txt',
    'sdxl_partiprompts.txt',
    'sdxl_coverage.txt',
]

# dummy for now
metadata = [
    {'metric': 'clip'},
    {'metric': 'pickscore'},
    {'metric': 'hpsv2'},
    {'metric': 'hpsv2'},
    {'coverage': True}
]

# path_pattern = r"kl(-?\d+\.?\d*)/.*?/saved_images/([^/]+)/"
path_pattern = r"kl([-+]?\d*\.?\d+)/.*saved_images/(.+)/"

def postprocess_method(method):
    if 'power' in method:
        return float(method.split("_")[1])
    elif method == 'linear':
        return 1.0
    elif method == 'draft':
        return 0
    elif method == 'base':
        return np.inf
    else:
        raise ValueError

# get name and kl
def find_method_name(data):
    # assert np.array(["/opt/nemo-aligner/" in line for line in data]).sum() == 1, data
    found = False
    klterm, method = 0, 0
    for line in data:
        m = re.search(path_pattern, line)
        if m:
            found = True
            klterm, method = m.group(1), m.group(2)
    assert found
    return float(klterm), method

def tryfloat(token):
    try:
        f = float(token)
        return f
    except:
        return None

def find_metrics(data, meta={}):
    ''' given lines of data, and optional metadata, find metrics '''
    res = {}
    if meta.get('metric'):
        cfg = res[meta['metric']] = {}
    else:
        cfg = res

    for line in data:
        tokens = line.replace("\t", " ").split(" ")
        is_num = np.array([int(isinstance(tryfloat(x), float)) for x in tokens])
        if np.sum(is_num) == 1:
            fidx = np.where(is_num)[0][0]
            val = float(tokens[fidx])
        elif np.sum(is_num) == 2:
            fidx, sidx = np.where(is_num)[0][:2]
            val = float(tokens[fidx])
        else:
            assert np.sum(is_num) == 0, line
            continue
        # get the part before the first part 
        prefix = " ".join(tokens[:fidx]).lower()
        # this is a coverage metric
        if meta.get('coverage', False):
            cfg[prefix.split(":")[0]] = val
        else:
            # this is not a coverage file, extract the dataset
            if 'parti' in prefix:
                cfg['PartiPrompt'] = val
            elif 'hps' in prefix:
                if 'anime' in prefix:
                    cfg['HPSv2 (Anime)'] = val
                elif 'concept' in prefix:
                    cfg['HPSv2 (Concept Art)'] = val
                elif 'paintings' in prefix:
                    cfg['HPSv2 (Paintings)'] = val
                elif 'photo' in prefix:
                    cfg['HPSv2 (Photo)'] = val
            else:
                raise ValueError
    return res

# compile results
results = defaultdict(dict)

for file, meta in zip(files, metadata):
    # load file 
    # if meta.get('metric') != 'hpsv2':
    #     continue
    with open(file, 'r') as fi:
        data = fi.read()
    datalines = list(filter(lambda x: len(x) > 0, data.split("\n\n")))
    # extract metrics too
    for data in datalines:
        data = data.split("\n")
        kl, method = find_method_name(data)
        method = postprocess_method(method)
        metrics = find_metrics(data, meta)
        for k, v in metrics.items():
            if isinstance(v, dict):
                if results[(kl, method)].get(k) is None:
                    results[(kl, method)][k] = {}
                results[(kl, method)][k].update(v)
            else:
                results[(kl, method)][k] = v

results_n = {str(k[0]) + "_" + str(k[1]): v for k, v in results.items()}

save_json = "sdxl_results.json"
if not osp.exists(save_json):
    with open(save_json, 'w') as fi:
        fi.write(json.dumps(results_n))
else:
    from pprint import pprint
    pprint(results_n)
    raise ValueError("Save path exists already - printing")