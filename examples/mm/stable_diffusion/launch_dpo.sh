#!/bin/bash
PROJECT="NeMo-SD-DPO"
WANDB="8256bec8f68d1a0ee4a3208685a8db0474d3806b"
 
export PYTHONPATH=/opt/NeMo:/opt/nemo-aligner:$PYTHONPATH

LR=${LR:=0.00001}
ACTOR_MICRO_BS=${ACTOR_MICRO_BS:=1}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:=2}
ACTOR_NUM_DEVICES=8

ACTOR_GLOBAL_BATCH_SIZE=$((ACTOR_MICRO_BS*ACTOR_NUM_DEVICES*GRAD_ACCUMULATION))
KL_COEF=0.1


ACTOR_CONFIG_PATH="/opt/nemo-aligner/examples/mm/stable_diffusion/conf"
ACTOR_CONFIG_NAME="sd_dpo"
ACTOR_CKPT="/opt/nemo-aligner/checkpoints/model_weights.ckpt"
VAE_CKPT="/opt/nemo-aligner/checkpoints/vae.bin"
ACTOR_WANDB_NAME=DPO-EOS-LR_${LR}-BS_${ACTOR_GLOBAL_BATCH_SIZE}
DIR_SAVE_CKPT_PATH="/opt/nemo-aligner/diff_dpo_saved_ckpts"
DATASET_PATH="/opt/nemo-aligner/datasets/animals45.tar"
LOG_DIR=/opt/nemo-aligner/examples/mm/stable_diffusion/logs_LR_${LR}_BS_${ACTOR_GLOBAL_BATCH_SIZE}

mkdir -p ${DIR_SAVE_CKPT_PATH}
mkdir -p ${LOG_DIR}

ACTOR_DEVICE="0,1,2,3,4,5,6,7"
echo "Running Diffusion DPO on ${ACTOR_DEVICE}"
wandb login ${WANDB} \
&& CUDA_VISIBLE_DEVICES="${ACTOR_DEVICE}" python /opt/nemo-aligner/examples/mm/stable_diffusion/train_sd_dpo.py \
    --config-path=${ACTOR_CONFIG_PATH} \
    --config-name=${ACTOR_CONFIG_NAME} \
    model.optim.lr=${LR} \
    model.optim.weight_decay=0.005 \
    model.optim.sched.warmup_steps=0 \
    model.infer.inference_steps=25 \
    model.infer.eta=0.0 \
    model.kl_coeff=${KL_COEF} \
    model.truncation_steps=1 \
    model.max_epochs=1 \
    model.max_steps=10000 \
    model.save_interval=500 \
    model.unet_config.from_pretrained=${ACTOR_CKPT} \
    model.first_stage_config.from_pretrained=${VAE_CKPT} \
    model.micro_batch_size=${ACTOR_MICRO_BS} \
    model.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE} \
    model.peft.enable=True \
    model.data.train.dataset_path=${DATASET_PATH} \
    model.data.webdataset.local_root_path=${DATASET_PATH} \
    trainer.val_check_interval=20 \
    trainer.gradient_clip_val=10.0 \
    trainer.devices=${ACTOR_NUM_DEVICES} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${ACTOR_WANDB_NAME} \
    exp_manager.resume_if_exists=True \
    exp_manager.explicit_log_dir=${DIR_SAVE_CKPT_PATH} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT} > ${LOG_DIR}/stdout_dpo_log_${SLURM_LOCALID}.txt 2> ${LOG_DIR}/stderr_dpo_log_${SLURM_LOCALID}.txt
