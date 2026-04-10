# IMAGE_TRAIN_SIZE refer to 81 token embeddings
# Please run download_models.py to download the required models to 'models/Nexus-GenV2' before training
# USE_IMG_EMBED_AS_INPUT=false refers to the prefilled autoregression strategy
PYTHONPATH=$(pwd) \
IMAGE_TRAIN_SIZE=252 \
USE_IMG_EMBED_AS_INPUT=false \
TOKEN_LOSS_WEIGHT=3.0 \
IMG_LOSS_WEIGHT=3.0 \
MAX_PIXELS=262640 \
nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model 'models/Nexus-GenV2' \
    --model_type 'qwenall2all_custom' \
    --template 'qwenall2all_custom' \
    --custom_register_path train/ar/model.py \
                           train/ar/template.py \
    --dataset 'assets/example_datasets/llm_dataset.jsonl' \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --freeze_vit true \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 5000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --max_length 4096 \
    --deepspeed zero3 \
