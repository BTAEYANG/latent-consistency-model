MODEL_DIR="/home/wq/Desktop/diffusers/stable-diffusion-v1-5"
OUTPUT_DIR="./lcm-lora-cifar10"

python train_lcm_distill_lora_sd_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --dataset_type=cifar10 \
    --dataset_path=./data \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=1e-6 \
    --loss_type="huber" \
    --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --dataloader_num_workers=4 \
    --validation_steps=100 \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=3 \
    --train_batch_size=8 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --seed=42 \
    --report_to=wandb