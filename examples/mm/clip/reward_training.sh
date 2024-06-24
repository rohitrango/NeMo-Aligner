## all other env variables are copied from before
export WANDB="d727288f26d60019e79de694e1a803181a18aab6"    
wandb login ${WANDB}
torchrun --nproc-per-node=8 $DISTRIBUTED_PARAMS --nnodes=$NNODES /opt/nemo-aligner/examples/mm/clip/train_reward_model.py \
    model.micro_batch_size=$MICRO_BATCH_SIZE trainer.devices=8 trainer.num_nodes=$NNODES exp_manager.create_wandb_logger=True $ADDITIONAL_KWARGS

