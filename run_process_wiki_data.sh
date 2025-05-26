module purge

# load environment
module load anaconda/3
conda activate honesty_env 

cd ~/research/QA_Exploration
python process_wiki_data.py