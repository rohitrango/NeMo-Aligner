import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DiffusionPipeline
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import DiffusionEngine
from omegaconf import OmegaConf
import torch
from functools import partial
from torch.nn import ModuleList

vae_hf_dir = "/opt/nemo-aligner/checkpoints/sdxl/vae"
unet_hf_dir = "/opt/nemo-aligner/checkpoints/sdxl/unet"

# vae_nemo_cfg = ""
nemo_cfg = OmegaConf.load("/opt/NeMo/examples/multimodal/text_to_image/stable_diffusion/conf/sd_xl_base.yaml")
vae_config = nemo_cfg.model.first_stage_config
vae_config.from_pretrained = "/opt/nemo-aligner/checkpoints/sdxl/vae_nemo.ckpt"
vae_config._target_ = "nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKLInferenceWrapper"
vae_config.from_NeMo = True

unet_config = nemo_cfg.model.unet_config
unet_config.from_pretrained = "/opt/nemo-aligner/checkpoints/sdxl/unet_nemo.ckpt" 
unet_config.from_NeMo = True

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder import AutoencoderKLInferenceWrapper

def compare_vae():
    vae_hf = AutoencoderKL.from_pretrained(vae_hf_dir).cuda()
    vae_nemo = DiffusionEngine.from_config_dict(vae_config).cuda()
    tensor = torch.randn(1, 4, 64, 64).cuda()
    # get outputs
    with torch.no_grad():
        hf_decoded = vae_hf.decode(tensor)['sample']
        nemo_decoded = vae_nemo.decode(tensor)
        print(torch.abs(hf_decoded - nemo_decoded).mean())
        print(torch.abs(hf_decoded - nemo_decoded).mean() / torch.abs(nemo_decoded).mean())
        print((torch.abs(hf_decoded - nemo_decoded)/(1e-8 + torch.abs(nemo_decoded))).mean())

def compare_unet():
    unet_hf = UNet2DConditionModel.from_pretrained(unet_hf_dir).cuda().eval()
    unet_nemo = DiffusionEngine.from_config_dict(unet_config).cuda().eval()

    # # this is all u need for nemo
    tensor = torch.randn(1, 4, 64, 64).cuda()
    time = torch.zeros(1,).cuda()
    y = torch.zeros(1, 2816).cuda()
    context = torch.zeros(1, 80, 2048).cuda()
    # extra context for hf
    add_cond_kwargs = {
        'time_ids': torch.zeros(1, 6).cuda(),
        'text_embeds': torch.zeros(1, 1280).cuda(),
    }

    # nemo_pre_hooks = {}
    # nemo_pre_hook_handles = {}

    # add forward hooks and a dict to save all the outs
    hf_debug, hf_handles = {}, {}
    nemo_debug, nemo_handles = {}, {}
    def hf_hook(key, model, input, output):
        print(output)
        hf_debug[key] = output
        return output
    
    def nemo_hook(key, model, input, output):
        print(output)
        nemo_debug[key] = output
        return output
    
    def getattr_recur(module, name):
        namesplit = name.split(".")
        mod = module
        for n in namesplit:
            mod = getattr(mod, n)
        return mod
    
    def clear_handles(dict):
        keys = list(dict.keys())
        for k in keys:
            dict[k].remove()
            del dict[k]
    
    clear_handles(hf_handles); clear_handles(nemo_handles)
    
    ##### HF
    # add hooks to zero out embeddings
    unet_hf.add_embedding.register_forward_hook(lambda model, input, output: torch.zeros_like(output))
    unet_hf.time_embedding.register_forward_hook(lambda model, input, output: torch.zeros_like(output))
    
    hf_keys = ['down_blocks', 'mid_block', 'up_blocks']
    for k in hf_keys:
        mod = getattr_recur(unet_hf, k)
        if isinstance(mod, ModuleList):
            mod = mod[-1]
        handle = mod.register_forward_hook(partial(hf_hook, k))
        hf_handles[k] = handle
    
    # run output
    with torch.no_grad():
        z_hf = unet_hf(tensor, time, context, added_cond_kwargs=add_cond_kwargs)['sample']


    #### Nemo
    unet_nemo.label_emb.register_forward_hook(lambda model, input, output: torch.zeros_like(output))
    unet_nemo.time_embed.register_forward_hook(lambda model, input, output: torch.zeros_like(output))
    nemo_keys = ['input_blocks', 'middle_block', 'output_blocks']
    for k in nemo_keys:
        handle = getattr_recur(unet_nemo, k)[-1].register_forward_hook(partial(nemo_hook, k))
        nemo_handles[k] = handle
    
    with torch.no_grad():
        z_nemo = unet_nemo(tensor, timesteps=time, y=y, context=context)

    def error(t1, t2, eps=1e-8):
        abs_error = torch.abs(t1 - t2).mean()
        rel_error = (torch.abs(t1 - t2)/(torch.abs(t2) + eps)).mean()
        return abs_error, rel_error
    
    print(error(nemo_debug['output_blocks'], hf_debug['up_blocks'][0]))
    print(error(nemo_debug['middle_block'], hf_debug['mid_block'][0]))
    print(error(nemo_debug['input_blocks'], hf_debug['down_blocks'][0]))
    print(error(z_nemo, z_hf))

    # y = contains the input to label_emb or add_emb (clip pooled features + HW embedding)
    # context contains CLIP VIT context (B, S, D)
    # with torch.no_grad():
    #     out = unet_nemo(tensor, timesteps=time, y=y, context=context)

    # write custom forward pass for hf model
    # self refers to hf model (for easier copying)
    # def hf_forward(self, tensor, timestep, y, context):
    #     t_emb = self.get_time_embed(sample=tensor, timestep=timestep)
    #     emb = self.time_embedding(t_emb, None)
    #     # aug_emb = self.get_aug_embed(
    #     #     emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    #     # )
    #     # aug_emb = torch.zeros_like(aug_emb)
    #     aug_emb = self.add_embedding(y)
    #     emb = emb + aug_emb
    #     if self.time_embed_act is not None:
    #         emb = self.time_embed_act(emb)
    #     # process sample 
    #     h = self.conv_in(tensor)
    #     for down in self.down_blocks:
    #         h = down(h, temb=emb, encoder_hidden_states=context)
    #     # mid block todo
    #     return h 
    pass


def compare_sdxl():
    ''' compare the full SDXL modules in HF and nemo '''
    nemo_net = DiffusionEngine.from_config_dict(nemo_cfg.model, trainer=None).cuda()
    # hf_net = DiffusionPipeline.from_pretrained("/opt/nemo-aligner/checkpoints/sdxl/", torch_dtype=torch.float32, use_safetensors=True)
    hf_net = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True)
    hf_net.to("cuda")

    # add forward hooks and a dict to save all the outs
    hf_debug, hf_handles = {}, {}
    nemo_debug, nemo_handles = {}, {}
    def hf_hook(key, model, input, output):
        hf_debug[key] = output
    
    def getattr_recur(module, name):
        namesplit = name.split(".")
        mod = module
        for n in namesplit:
            mod = getattr(mod, n)
        return mod
    
    hf_keys = ['add_embedding.linear_2', 'down_blocks', 'time_embedding.linear_2']
    for k in hf_keys:
        handle = getattr_recur(hf_net.unet, k).register_forward_hook(partial(hf_hook, k))
        hf_handles[k] = handle


if __name__ == '__main__':
    compare_vae()
    compare_unet()