#!/usr/bin/bash
echo "Starting Job"
train_file=$1
export PATH="/users/sista/kkontras/anaconda3/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gl_env
echo $train_file
which python
python3 -V
cd /users/sista/kkontras/Documents/Sleep_Project
echo $PWD
echo $CUDA_VISIBLE_DEVICES
python3 $train_file
echo "Job finished"

