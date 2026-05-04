#!/bin/bash
#SBATCH --job-name=bc_llama
#SBATCH --account=hitz-exclusive
#SBATCH --partition=hitz-exclusive
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=99GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/evalero/projects/merge_llm/merges/logs/out/bc/llama/%A_%a.out
#SBATCH --error=/scratch/evalero/projects/merge_llm/merges/logs/err/bc/llama/%A_%a.err

module load Miniforge3
source /scratch/evalero/envs/merge_env/bin/activate

export HF_HOME="/scratch/evalero/huggingface_cache"
export HF_TOKEN="${HF_TOKEN}"

OUTPUT_DIR="/scratch/evalero/projects/merge_llm/merges/models/bc/llama/cat_Llama-3.1-8B__Llama-3.1-8B-Instruct_${SLURM_JOB_ID}"

mergekit-yaml /scratch/evalero/projects/merge_llm/merges/scripts/bc/llama/cat_Llama-3.1-8B__Llama-3.1-8B-Instruct.yaml \
  "$OUTPUT_DIR" \
  --copy-tokenizer \
  --cuda \
  --low-cpu-memory \
  --out-shard-size 5B \
  --lazy-unpickle \
  --allow-crimes

JSON_CONFIG_SRC="/scratch/evalero/json_configs/Llama-3.1"
cp "${JSON_CONFIG_SRC}"/*.json "${OUTPUT_DIR}/"