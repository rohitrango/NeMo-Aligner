#!/bin/bash
PROJECT="NeMo-draft+"
WANDB="d727288f26d60019e79de694e1a803181a18aab6"
export WANDB_ENTITY="nvidia"

export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

ACTOR_NUM_DEVICES=2
ACTOR_MICRO_BS=1
GRAD_ACCUMULATION=4
ACTOR_GLOBAL_BATCH_SIZE=$((ACTOR_MICRO_BS*ACTOR_NUM_DEVICES*GRAD_ACCUMULATION))
KL_COEF=0.1
LR=0.0005

ACTOR_CONFIG_PATH="/opt/nemo-aligner/examples/mm/stable_diffusion/conf"
ACTOR_CONFIG_NAME="draftp_sd"
ACTOR_CKPT="/opt/nemo-aligner/checkpoints/model_weights.ckpt"
VAE_CKPT="/opt/nemo-aligner/checkpoints/vae.bin"
# RM_CKPT="/home/ataghibakhsh/converted_pickscore.nemo"
RM_CKPT="/opt/nemo-aligner/checkpoints/pickscore.nemo"
ACTOR_WANDB_NAME=DRaFT+--ws-LR_${LR}-KL_${KL_COEF}-BS_${ACTOR_GLOBAL_BATCH_SIZE}
DIR_SAVE_CKPT_PATH="/opt/nemo-aligner/draft_p_saved_ckpts"
# DATASET_PATH="/opt/nemo-aligner/datasets/pickapic_51306.tar"
DATASET_PATH="/opt/nemo-aligner/datasets/pickapic50k.tar"

mkdir -p ${DIR_SAVE_CKPT_PATH}

ACTOR_DEVICE="0,1"
echo "Running DRaFT on ${ACTOR_DEVICE}"
git config --global --add safe.directory /opt/nemo-aligner \
&& wandb login ${WANDB} \
&& MASTER_PORT=15003 CUDA_VISIBLE_DEVICES="${ACTOR_DEVICE}" torchrun --nproc_per_node=${ACTOR_NUM_DEVICES} /opt/nemo-aligner/examples/mm/stable_diffusion/train_sd_draftp.py \
    --config-path=${ACTOR_CONFIG_PATH} \
    --config-name=${ACTOR_CONFIG_NAME} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=0.005 \
    model.optim.sched.warmup_steps=0 \
    model.infer.inference_steps=25 \
    model.infer.eta=0.0 \
    model.kl_coeff=${KL_COEF} \
    model.truncation_steps=1 \
    trainer.draftp_sd.max_epochs=1 \
    trainer.draftp_sd.max_steps=4000 \
    trainer.draftp_sd.save_interval=500 \
    model.unet_config.from_pretrained=${ACTOR_CKPT} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.micro_batch_size=${ACTOR_MICRO_BS} \
    model.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    model.peft.peft_scheme="none" \
    model.data.webdataset.local_root_path=${DATASET_PATH} \
    rm.model.restore_from_path=${RM_CKPT} \
    trainer.draftp_sd.val_check_interval=1 \
    trainer.draftp_sd.limit_val_batches=1 \
    trainer.draftp_sd.gradient_clip_val=10.0 \
    trainer.devices=${ACTOR_NUM_DEVICES} \
    rm.trainer.devices=${ACTOR_NUM_DEVICES} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_WANDB_NAME} \
    exp_manager.resume_if_exists=False \
    exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} 