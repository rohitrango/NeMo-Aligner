# this is for sdxl draft+
N=1
set -x
# for kl in 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 ; do
# 	sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:default_sdxl_kl${kl} --export=ALL,KL_COEF=${kl},JOBNAME=sdxl_draft_mainrun_kl${kl} launch_draft_eos_xl.sh
# done

# for kl in 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 ; do
# 	# echo sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:default_sd_kl${kl} --export=ALL,KL_COEF=${kl},JOBNAME=sd_draft_mainrun_kl${kl} launch_draft_eos.sh
# 	sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:default_sd_kl${kl} --export=ALL,KL_COEF=${kl},JOBNAME=sd_draft_mainrun_kl${kl} launch_draft_eos.sh
# done

##### this is for HPSv2 model

# for kl in 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 ; do
# 	sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:hpsv2_sdxl_kl${kl} --export=ALL,RM_CKPT="/opt/nemo-aligner/checkpoints/hpsv2/hpsv2_model.nemo",KL_COEF=${kl},JOBNAME=sdxl_draft_hpsv2_kl${kl} launch_draft_eos_xl.sh
# done

for kl in 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 ; do
	sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:hpsv2_sd_kl${kl} --export=ALL,RM_CKPT="/opt/nemo-aligner/checkpoints/hpsv2/hpsv2_model.nemo",KL_COEF=${kl},JOBNAME=sd_draft_hpsv2_kl${kl} launch_draft_eos.sh
done