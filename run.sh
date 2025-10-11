#!/bin/bash

# 1. Activate conda environment
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tsf-ad

# # 2. Loop through files in data/raw_data/ starting with "bs"
# for file in data/raw_data/bs*; do
#     # 3. Run Python file for each file
#     echo "Processing $file..."
#     python data/DataProcess_bs.py "$file"
# done

# 3. Loop through files in data/raw_data/ starting with "dg"
for file in data/raw_data/dg*; do
    # 3. Run Python file for each file
    echo "Processing $file..."
    python data/DataProcess_dg.py "$file"
done



