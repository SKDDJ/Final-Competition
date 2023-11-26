#!/bin/bash

# 定义输出文件名
output_load_model="load_model_output.txt"
output_score="score_output.txt"
total_time="total_time.txt"
combined_output="combined_output.txt"

export PATH="/home/schengwei/anaconda3/envs/ldm/bin:$PATH"

# export CUDA_VISIBLE_DEVICES="6"
# unset CUDA_VISIBLE_DEVICES

# 初始化并激活 Conda 环境
# 替换 YOUR_CONDA_PATH 为您的 conda 安装路径
# 替换 YOUR_ENV_NAME 为您的 conda 环境名
source /home/schengwei/anaconda3/etc/profile.d/conda.sh && conda activate ldm

# 记录 load_model.py 的开始和结束时间
start_load_model=$(date +%s)
/home/schengwei/anaconda3/envs/ldm/bin/accelerate launch --mixed_precision fp16 --num_processes 1 load_model.py > "$output_load_model"
end_load_model=$(date +%s)

# 计算并记录运行时间
echo "Runtime for load_model.py: $((end_load_model - start_load_model)) seconds" >> "$total_time"

# 记录 score.py 的开始和结束时间
start_score=$(date +%s)
/home/schengwei/anaconda3/envs/ldm/bin/python score.py > "$output_score"
end_score=$(date +%s)

# 计算并记录运行时间
echo "Runtime for score.py: $((end_score - start_score)) seconds" >> "$total_time"

# 合并输出文件
cat "$total_time" "$output_score" "$output_load_model" > "$combined_output"

# 发送合并后的文件到您的邮箱
# 注意：请将 YOUR_EMAIL_ADDRESS 替换成您的邮箱地址
mail -s "Combined Output and Runtime of Scripts" yimingshi666@gmail.com < "$combined_output"
