#! /bin/bash
# Read arguments from command line
model_name_or_path=$1
revision=$2
train_file=$3
validation_file=$4
num_train_epochs=$5
learning_rate=$6
job_id=$7
tasks=$8
max_gen_toks=$9

# Extract the last part of the train_file path
train_file_name=$(basename ${train_file})
model_name=$(basename ${model_name_or_path})

# Create a RUN_ID based on model_name_or_path, train_file, num_train_epochs, and learning_rate
export WANDB_RUN_ID="${model_name}-${train_file_name}-epochs-${num_train_epochs}-lr-${learning_rate}"

# Setting up other environment variables
echo "Running on $(hostname)"
echo "Learning rate ${learning_rate}"
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_PROJECT=finetuned
export WANDB_RESUME=allow
export ABS_PATH=.../LLMindCraft
export PYTHONPATH="$ABS_PATH"
export MONGO_URI=""
export HF_HOME="$ABS_PATH/saved_models"
export HF_TOKEN=''

# Set up directories
output_dir="./output_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}
cache_dir="./cache_models"
mkdir -p ${cache_dir}
cutoff_len=4096


# Training command
torchrun --nproc_per_node 4 src/ft/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --revision ${revision} \
    --server_id $(hostname) \
    --job_id ${job_id} \
    --tasks ${tasks} \
    --max_gen_toks ${max_gen_toks} \
    --llama \
    --use_lora \
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs ${num_train_epochs} \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate ${learning_rate} \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir
