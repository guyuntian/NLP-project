python main.py \
    --output_dir outputs/finetune/ \
    --learning_rate 2e-6 \
    --model_name_or_path outputs/pretrain \
    --training_steps 10 \
    --head_lr 1e-4 \
    --neg_sample 100