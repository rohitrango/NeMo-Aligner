# this is for sdxl draft+
N=1
for kl in 0.0 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 ; do
	echo sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:default_sdxl_kl${kl} --export=ALL,JOBNAME=sdxl_draft_mainrun_kl${kl} launch_draft_eos_xl.sh
	sbatch -A coreai_dlalgo_llm -t 04:00:00  --array=1-$N%1 --job-name=coreai_dlalgo_llm-draftp:default_sdxl_kl${kl} --export=ALL,JOBNAME=sdxl_draft_mainrun_kl${kl} launch_draft_eos_xl.sh
done
