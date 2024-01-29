PROJECT_PATH=/orange/yonghui.wu/qx68/platform/LLMindCraft
export HF_HOME=""
export HF_TOKEN=''
export PYTHONPATH=$PROJECT_PATH

python src/merge_llama_with_lora.py \
    --model_name_or_path chaoyi-wu/MedLLaMA_13B \
    --lora_path saved_models/finetuned_plos_pmc/checkpoint-9792/ \
    --output_path saved_models/merged_plos \
    --llama
