#!/bin/bash
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_sim.json --output_path outputs/boy1_sim --weight_dir model_output/boy1
python sample.py --restore_path model_output/boy2 --prompt_path eval_prompts_advance/boy2_sim.json --output_path outputs/boy2_sim --weight_dir model_output/boy2
python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_sim.json --output_path outputs/girl1_sim  --weight_dir model_output/girl1
python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_sim.json --output_path outputs/girl2_sim  --weight_dir model_output/girl2
python sample.py --restore_path model_output/boy1 --prompt_path eval_prompts_advance/boy1_edit.json --output_path outputs/boy1_edit   --weight_dir model_output/boy1
python sample.py --restore_path model_output/boy2  --prompt_path eval_prompts_advance/boy2_edit.json --output_path outputs/boy2_edit  --weight_dir model_output/boy2
python sample.py --restore_path model_output/girl1 --prompt_path eval_prompts_advance/girl1_edit.json --output_path outputs/girl1_edit  --weight_dir model_output/girl1
python sample.py --restore_path model_output/girl2 --prompt_path eval_prompts_advance/girl2_edit.json --output_path outputs/girl2_edit    --weight_dir model_output/girl2


