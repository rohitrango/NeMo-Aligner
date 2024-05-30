import torch
import numpy as np
import json
from pprint import pprint

def filter_keys(rule, dict):
    keys = list(dict.keys())
    nd = {k: dict[k] for k in keys if rule(k)}
    return nd

def map_keys(rule, dict):
    new = {rule(k): v for k, v in dict.items()}
    return new

def printlist(key_or_dict, alphabetic=False):
    if isinstance(key_or_dict, dict):
        sortedkeys = sorted(key_or_dict.keys(), key=lambda t: key_or_dict[t].shape if not alphabetic else t)
        for k in sortedkeys:
            print(k, key_or_dict[k].shape)
    else:
        keys = sorted(list(keys))
        for k in keys:
            print(k)

def nelements(dict, keys=None):
    if keys is None:
        keys = dict.keys() 
    n = np.sum([np.prod(list(dict[k].shape)) for k in keys])
    return n

def create_groups(state_dict, ):
    # create groups and assign them their sizes
    prefix = set(map(lambda x: x.split(".")[0], list(state_dict.keys())))
    dict = {}
    for k in prefix:
        groupk = filter(lambda x: x.split(".")[0] == k, list(state_dict.keys()))
        dict[k] = nelements(state_dict, list(groupk))
    return dict

# HF to nemo mapping
# unet_mapping = {
#     "up_blocks": "output_blocks",
#     "time_embedding": "time_embed",
#     "mid_block": "middle_block",
#     ("conv_norm_out", "conv_out"): "out",
#     "add_embedding": "label_emb",
#     ("conv_in", "down_blocks"): "input_blocks"
# }

def tolist(k):
    if not isinstance(k, (list, tuple)):
        return [k]
    return list(k)

############################################################################################################################################################

# nemo_path = "/opt/nemo-aligner/draftp_xl_saved_ckpts/checkpoints/sdxl_draftp_train--reduced_train_loss=0.00-step=1-consumed_samples=2-last.ckpt"


# hf_unet = torch.load(hf_unet_path)

# print("nemo...")
# pprint(create_groups(nemo_data), indent=4)

# print("hf...")
# pprint(create_groups(hf_unet), indent=4)

# # nelements_unet = np.sum([np.prod(list(x.shape)) for x in nemo_data.values()])
# # nelements_unet_hf = np.sum([np.prod(list(x.shape)) for x in hf_unet_path.values()])
# # print(nelements_unet, nelements_unet_hf)

# # printlist(nemo_data)
# # print("...hf...")
# # printlist(hf_unet_path)

# for hfkey, nemokey in unet_mapping.items():
#     hfkey, nemokey = tolist(hfkey), tolist(nemokey)
#     hf_filt_dict = filter_keys(lambda k: any([k.startswith(x + ".") for x in hfkey]), hf_unet)
#     nemo_filt_dict = filter_keys(lambda k: any([k.startswith(x + ".") for x in nemokey]), nemo_data)
#     print("...hf...")
#     printlist(hf_filt_dict, alphabetic=True)
#     print("...nemo...")
#     printlist(nemo_filt_dict, alphabetic=True)
