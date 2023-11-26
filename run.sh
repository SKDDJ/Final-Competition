#!/bin/bash


# This script runs a Docker container with specified GPU device and loads a tar file containing the competition code.
# The script then creates directories for output files and runs the Docker container with specified volumes and entrypoint.

# Get the tar file and GPU device from command line arguments.
tar_file=$1
device=$2

# Set the paths for the input JSON data and the output bound JSON.
json_data_path=/home/final_evaluation/test_data
bound_json_path=/home/final_evaluation/bound_json

# Print the current dataset and bound JSON paths.
echo --- test script of final competition of lora --
echo curring dataset: $json_data_path
echo bound json: $bound_json_path

# Enable debugging mode.
set -x

# Load the Docker image from the tar file and get its identifier.
# ident=`docker load --input $tar_file | python /home/PZDS/get_indent.py`
ident=$tar_file
echo loaded: $ident

# Get the MD5 hash of the tar file identifier.
file_ident=`echo $ident | python -c "import hashlib; print(hashlib.md5(input().encode('utf-8')).hexdigest());"`
echo file_ident: $file_ident

# Create directories for output files.
mkdir -p /home/final_jsons/$file_ident
mkdir -p /home/final_images/$file_ident
mkdir -p /home/final_scores/$file_ident

# Run the Docker container with specified volumes and entrypoint.
docker run -it --gpus "device=${device}" --shm-size=4g  --rm \
-v /home/PZDS/common_insightface:/root/.insightface:ro \
-v /home/final_evaluation/ImageReward:/workspace/ImageReward:ro \
-v /home/final_evaluation/models--bert-base-uncased:/root/.cache/huggingface/hub/models--bert-base-uncased:ro \
-v /home/PZDS/models:/workspace/models:ro \
-v /home/PZDS/common_diffusers_model/unidiffuser_hf:/workspace/diffusers_models:ro \
-v /home/final_evaluation/indocker_shell.sh:/workspace/indocker_shell.sh:ro \
-v /home/train_outputs/$file_ident.log:/workspace/train_out.log \
-v $json_data_path:/workspace/final_json_data:ro \
-v $bound_json_path:/workspace/bound_json_outputs:ro \
-v /home/final_jsons/$file_ident:/workspace/json_outputs \
-v /home/final_images/$file_ident:/workspace/image_outputs \
-v /home/final_scores/$file_ident:/workspace/score_outputs \
-v /home/final_evaluation/main.py:/workspace/main.py:ro \
-v /home/final_evaluation/baseline1.py:/workspace/load_model.py:ro \
--entrypoint /bin/bash $ident

### the old one 
# tar_file=$1
# device=$2

# sha256=`docker load --input $tar_file | grep -Po "sha256:(\w+)" | sed 's/sha256:\(.*\)/\1/g'`

# # docker run -it --gpus "device=${device}" --rm -v /home/schengwei/Competitionrepo/models:/workspace/models   -v /home/schengwei/.cache:/root/.cache $sha256

# # -v /root/indocker_shell.sh:/workspace/indocker_shell.sh $sha256
# docker run -it --gpus "device=${device}" --rm -v /home/test01/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/test01/train_data:/workspace/train_data -v /home/test01/models:/workspace/models \



# # docker run -it --gpus "device=${device}" --rm -v /home/wuyujia/competition/eval_prompts_advance:/workspace/eval_prompts_advance 
# # -v /home/wuyujia/competition/train_data:/workspace/train_data -v /home/wuyujia/competition/models:/workspace/models \
# # -v /home/wuyujia/competition/indocker_shell.sh:/workspace/indocker_shell.sh $sha256 

# # sudo docker run -it --gpus all --rm -v /home/wuyujia/competition/eval_prompts_advance:/workspace/eval_prompts_advance -v /home/wuyujia/competition/train_data:/workspace/train_data -v /home/wuyujia/competition/models:/workspace/models -v /home/wuyujia/competition/indocker_shell.sh:/workspace/indocker_shell.sh  -v /home/wuyujia/competition/sample.py:/workspace/sample.py -v /home/wuyujia/.insightface:/root/.insightface -v /home/wuyujia/.cache/huggingface:/root/.cache/huggingface xiugou:v1


# # sudo docker cp b012d72bdadd:/workspace /home/wuyujia/competition
