# hacky script to convert a ckpt file to a nemo file

from omegaconf import OmegaConf, open_dict
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import get_reward_model
import argparse
from hydra import initialize, compose
import argparse
from glob import glob
import os.path as osp
import re

def extract_metrics(filename, keys=['val_loss', 'val_accuracy']):
    # Create a regex pattern to match the keys and their values
    pattern = r'-(' + '|'.join(keys) + r')=([\d.]+)'
    # Find all matches in the filename
    matches = re.findall(pattern, filename)
    # Convert matches to a dictionary
    metrics = {key: float(value) for key, value in matches}
    return metrics

# Example usage
# filename = 'megatron_clip--val_loss=0.54--val_accuracy=0.6715-step=3200-consumed_samples=3276800.0.ckpt'
# keys = ['val_loss', 'val_accurcacy', 'step', 'consumed_samples']
# metrics = extract_metrics(filename, keys)
# print(metrics)
# exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--cfg_path", type=str, required=True)
    args = parser.parse_args()
    with initialize(version_base=None, config_path="./conf", job_name="test"):
        cfg = compose(config_name=args.cfg_path)
        #cfg = compose(config_name="transformer-4layer-frozen.yaml")
    # with initialize(version_base=None, config_path=args.path.replace("/opt/nemo-aligner/", "../../../"), job_name="test"):
        # cfg = compose(config_name="hparams.yaml")
    
    checkpoints = glob(osp.join(args.path, "checkpoints", "*.ckpt"))
    metrics = extract_metrics(checkpoints[0])
    if metrics.get('val_accuracy') is not None:
        ckpt = sorted(checkpoints, key=lambda x: extract_metrics(x)['val_accuracy'])[-1]  # choose highest acc
        print("Choosing highest acc model", ckpt)
    else:
        ckpt = sorted(checkpoints, key=lambda x: extract_metrics(x)['val_loss'])[0]  # choose lowest loss
        print("Choosing lowest loss model", ckpt)
    restore_from_path = ckpt

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.restore_from_path = restore_from_path
        
    model = get_reward_model(cfg, cfg.model.micro_batch_size, cfg.model.global_batch_size).eval()
    model.save_to(osp.join(args.path, "checkpoints", "pickscore_multicrop.nemo"))
    print(f"Saved to ", osp.join(args.path, "checkpoints", "pickscore_multicrop.nemo"))

    #model.save_to("/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr/checkpoints/pickscore_multicrop.nemo")
    # model.save_to("/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr_sbd/checkpoints/pickscore_multicrop.nemo")

