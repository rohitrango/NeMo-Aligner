# hacky script to convert a ckpt file to a nemo file

from omegaconf import OmegaConf, open_dict
from nemo_aligner.models.mm.stable_diffusion.image_text_rms import get_reward_model
import argparse
from hydra import initialize, compose

if __name__ == '__main__':
    with initialize(version_base=None, config_path="./conf", job_name="test"):
        cfg = compose(config_name="transformer-4layer-frozen.yaml")
    #restore_from_path = "/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr/checkpoints/megatron_clip--val_loss=0.52-step=4000-consumed_samples=512000.0.ckpt"
    restore_from_path = "/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr_sbd/checkpoints/megatron_clip--val_loss=0.52-step=4000-consumed_samples=512000.0.ckpt"
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.restore_from_path = restore_from_path
    model = get_reward_model(cfg, cfg.model.micro_batch_size, cfg.model.global_batch_size).eval()
    #model.save_to("/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr/checkpoints/pickscore_multicrop.nemo")
    model.save_to("/opt/nemo-aligner/checkpoints/multicrop-rm/t4layer_smalllr_sbd/checkpoints/pickscore_multicrop.nemo")

