
CUDA_VISIBLE_DEVICES=1 python3 ../source/token-classification.py \
  --model_name_or_path microsoft/deberta-v3-xsmall \
  --data_dir ../dataset/pos_ner \
  --output_dir ../tmp/ner_deberta \
  --do_train \
  --do_eval
