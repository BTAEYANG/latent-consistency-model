# Latent Consistency Distillation Example:

[Latent Consistency Models (LCMs)](https://arxiv.org/abs/2310.04378) is method to distill latent diffusion model to enable swift inference with minimal steps. This example demonstrates how to use the latent consistency distillation to distill stable-diffusion-v1.5 for less timestep inference.

## Reference to Hugging Face 🤗 Diffusers examples.

You can also refer to the official implementation of LCM from huggingface diffusers, see [here](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)

## Full model distillation

### Running locally with PyTorch

#### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups.


#### Example with LAION-A6+ dataset

```bash
runwayml/stable-diffusion-v1-5
PROGRAM="train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url='pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -' \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub \
```

## LCM-LoRA

Instead of fine-tuning the full model, we can also just train a LoRA that can be injected into any SDXL model.

### Example with LAION-A6+ dataset
    
```bash
runwayml/stable-diffusion-v1-5
PROGRAM="train_lcm_distill_lora_sd_wds.py \
    --pretrained_teacher_model=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --lora_rank=64 \
    --learning_rate=1e-6 --loss_type="huber" --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url='pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -' \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub \
```

### Example with CIFAR-10

```bash
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
```