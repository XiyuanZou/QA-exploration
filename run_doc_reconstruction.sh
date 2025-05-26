module purge

# load environment
module load anaconda/3
conda activate honesty_env 

cd ~/research/QA_Exploration
python doc_reconstruction_trial_and_error_prompt_method.py