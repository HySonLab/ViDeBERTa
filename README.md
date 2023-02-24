# ViDeBERTa: A powerful pre-trained language model for Vietnamese, EACL 2023

Paper: https://arxiv.org/pdf/2301.10439.pdf

## Contributors
* Tran Cong Dao
* Pham Nhut Huy
* Nguyen Tuan Anh
* Hy Truong Son (Correspondent / PI)

## Main components
1. [Pre-training](#pretraining)
2. [Model](#videberta)
3. [Fine-tuning](#finetuning)

## <a name="pretraining"></a> Pre-training
## Code architecture
1. bash: bash scripts to run the pipeline
2. config: model_config (json files)
3. dataset: datasets folder (both store original **txt** dataset and the pointer to memory of datasets.load_from_disk)
4. source: main python files to run pre-training tokenizers
5. tokenizer: folder to store tokenizers
### Pre-tokenizer
- Split the original **txt** datasets into train, validation and test sets with 90%, 5%, 5%.
- Using the PyVi library to segment the datasets
- Save datasets to disk
### Pre-train_tokenizer
- Load datasets
- Train the tokenizers with SentencePiece models
- Save tokenizers 
### Pre-train_model
- Load datasets
- Load tokenizers
- Pre-train DeBERTa-v3
## <a name="videberta"></a> Model
## <a name="finetuning"></a> Fine-tuning
### Code architecture
1. POS tagging and NER (POS_NER)
2. Question Answering (QA and QA2)
3. Open-domain Question Answering (OPQA)
