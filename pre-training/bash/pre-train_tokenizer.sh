CUDA_VISIBLE_DEVICES=0 python ../source/pre-train_tokenizer.py \
    --tokenizer_name microsoft/deberta-v3-xsmall \
    --source_dir /media/data/huypn10/Vietnamese%20Pretrained%20Model/pre-training/dataset/segment_cc100_1e4 \
    --destination_dir ../tokenizer/test \
    --cache_dir ../spm_cc100_slow