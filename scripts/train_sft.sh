accelerate launch --config_file /data/lzj/LLaMA-Efficient-Tuning/scripts/accelerate.yaml src/train_bash.py \
    --stage sft \
    --report_to wandb \
    --model_name_or_path /data/lzj/LLaMA-Efficient-Tuning/Baichuan-13B-Chat \
    --do_train \
    --prompt_template baichuan \
    --dataset step1_sft \
    --finetuning_type lora \
    --output_dir modeltest0731_sft \
    --lora_target W_pack,o_proj,gate_proj,up_proj,down_proj \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant \
    --lora_rank 8 \
    --lora_alpha 32 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 20.0 \
    --max_grad_norm 0.5 \
    --plot_loss \
    --fp16


    # --checkpoint_dir /data/lzj/LLaMA-Efficient-Tuning/modeltest0724/checkpoint-600 \
