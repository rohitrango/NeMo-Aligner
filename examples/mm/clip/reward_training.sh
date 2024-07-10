## all other env variables are copied from before
export WANDB="d727288f26d60019e79de694e1a803181a18aab6"    
wandb login ${WANDB}
export CONFIG_NAME=${CONFIG_NAME:="megatron_multicrop_rm_gap"}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:=16}
export NNODES=${NNODES:=1}
export JOB_NAME=${JOB_NAME=:="dummy"}
export LOG_WANDB=${LOG_WANDB:-"True"}

pip install decord

torchrun --nproc-per-node=8 $DISTRIBUTED_PARAMS --nnodes=$NNODES /opt/nemo-aligner/examples/mm/clip/train_reward_model.py \
    --config-name $CONFIG_NAME \
    model.micro_batch_size=$MICRO_BATCH_SIZE trainer.devices=8 trainer.num_nodes=$NNODES exp_manager.create_wandb_logger=${LOG_WANDB} \
    exp_manager.explicit_log_dir=/opt/nemo-aligner/checkpoints/multicrop-rm/${JOB_NAME} name=${JOB_NAME} $ADDITIONAL_KWARGS

