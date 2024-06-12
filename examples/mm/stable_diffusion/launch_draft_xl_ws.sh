#!/bin/bash
# this is meant to be an interactive node, so only 8 devices per expt
export PROJECT="NeMo-draft+"
export WANDB="d727288f26d60019e79de694e1a803181a18aab6"
export WANDB_ENTITY="nvidia"
export WANDB_API_KEY=${WANDB}
 
export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

LR=${LR:=0.00025}
INF_STEPS=${INF_STEPS:=25}
KL_COEF=${KL_COEF:=0.1}
ETA=${ETA:=0.0}
DATASET=${DATASET:="pickapic50k.tar"}
MICRO_BS=${MICRO_BS:=1}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=4}
PEFT=${PEFT:="sdlora"}
NUM_DEVICES=${NUM_DEVICES:=2}
DEFAULT_GBS=$((MICRO_BS*NUM_DEVICES*GRAD_ACCUMULATION))
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:=$DEFAULT_GBS}
LOG_WANDB=${LOG_WANDB:="True"}
JOBNAME=${JOBNAME:="dummy"}
SLEEP=${SLEEP:=0}

RUN_DIR=/opt/nemo-aligner/sdxl_draft_runs/sdxl_draft_run_${JOBNAME}_lr_${LR}_data_${DATASET}_kl_${KL_COEF}_bs_${GLOBAL_BATCH_SIZE}_infstep_${INF_STEPS}_eta_${ETA}_peft_${PEFT}
WANDB_NAME=SDXL_DRaFT+${JOBNAME}_lr_${LR}_data_${DATASET}_kl_${KL_COEF}_bs_${GLOBAL_BATCH_SIZE}_infstep_${INF_STEPS}_eta_${ETA}_peft_${PEFT}
WEBDATASET_PATH=/opt/nemo-aligner/datasets/${DATASET}

LOGDIR=${RUN_DIR}/logs
mkdir -p ${LOGDIR}
# UNET_CKPT="/opt/nemo-aligner/checkpoints/sdxl_ao_tang/unet.ckpt"
# VAE_CKPT="/opt/nemo-aligner/checkpoints/sdxl_ao_tang/vae.ckpt"

CONFIG_PATH="/opt/nemo-aligner/examples/mm/stable_diffusion/conf"
CONFIG_NAME=${CONFIG_NAME:="draftp_sdxl"}
UNET_CKPT="/opt/nemo-aligner/checkpoints/sdxl/unet_nemo.ckpt"
VAE_CKPT="/opt/nemo-aligner/checkpoints/sdxl/vae_nemo.ckpt"
RM_CKPT="/opt/nemo-aligner/checkpoints/pickscore.nemo"
DIR_SAVE_CKPT_PATH=/opt/nemo-aligner/sdxl_draft_runs/draftp_xl_saved_ckpts_${JOBNAME}
# NEMO_FILE="/opt/nemo-aligner/checkpoints/sdxl_base.nemo"      # this model is from Ao Tang

mkdir -p ${DIR_SAVE_CKPT_PATH}

# sleep $SLEEP
export DEVICE="0,1" && echo "Running DRaFT on ${DEVICE}"  && wandb login ${WANDB} && export HYDRA_FULL_ERROR=1 && CUDA_VISIBLE_DEVICES="${DEVICE}" torchrun --nproc_per_node=$NUM_DEVICES --master_port=30030 /opt/nemo-aligner/examples/mm/stable_diffusion/train_sdxl_draftp.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=0.0005 \
    model.optim.sched.warmup_steps=0 \
    model.sampling.base.steps=${INF_STEPS} \
    model.kl_coeff=${KL_COEF} \
    model.fsdp=True \
    model.truncation_steps=1 \
    trainer.draftp_sd.max_epochs=5 \
    trainer.draftp_sd.max_steps=10000 \
    trainer.draftp_sd.save_interval=200 \
    trainer.draftp_sd.val_check_interval=20 \
    trainer.draftp_sd.gradient_clip_val=10.0 \
    model.micro_batch_size=${MICRO_BS} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.peft.peft_scheme=${PEFT} \
    model.data.webdataset.local_root_path=$WEBDATASET_PATH \
    rm.model.restore_from_path=${RM_CKPT} \
    trainer.devices=${NUM_DEVICES} \
    rm.trainer.devices=${NUM_DEVICES} \
    exp_manager.create_wandb_logger=${LOG_WANDB} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.first_stage_config.from_NeMo=True \
    model.unet_config.from_pretrained=${UNET_CKPT} \
    model.unet_config.from_NeMo=True \
    exp_manager.wandb_logger_kwargs.name=${WANDB_NAME} \
    exp_manager.resume_if_exists=True \
    exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} # &> ${LOGDIR}/draft_log_${SLURM_LOCALID}.txt

