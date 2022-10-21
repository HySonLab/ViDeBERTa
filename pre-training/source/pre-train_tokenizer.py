import datasets
import argparse
from transformers import AutoTokenizer

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Tokenizer name')
    parser.add_argument('--source_dir', type=str, default=None, help='Source file')
    parser.add_argument('--destination_dir', type=str, default=None, help='Cache dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    raw_datasets = datasets.load_from_disk(args.source_dir)
    print(raw_datasets)
    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 10000):
            samples = dataset[start_idx : start_idx + 10000]
            yield samples["text"]

    training_corpus = get_training_corpus()
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_name)

    tokenizer = tokenizer.train_new_from_iterator(
        training_corpus,
        vocab_size=128000,
        min_frequency=5,
        show_progress=True,
    )
    # raw_datasets.save_to_disk("../dataset/saved_dataset/")
    tokenizer.save_pretrained(args.destination_dir)
    
