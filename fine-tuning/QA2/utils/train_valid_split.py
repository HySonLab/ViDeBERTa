import datasets
from datasets import concatenate_datasets
import glob

# datasets.temp_seed(101)
datasets.disable_progress_bar()


def assert_sample(sample):
    assert sample['context'][sample['answer_start_idx']: sample['answer_start_idx'] + len(sample['answer_text'])] == \
           sample['answer_text'], sample
    assert len(sample['context']) > 0
    assert len(sample['question']) > 0
    return True


def format_sample(sample):
    context_prev = sample['context'][:sample['answer_start_idx']].split()
    sample['answer_word_start_idx'] = len(context_prev)
    sample['answer_word_end_idx'] = len(context_prev) + len(sample['answer_text'].split()) - 1
    return sample


if __name__ == "__main__":
    train_set = []
    valid_set = []
    test_set = []
    dataset = datasets.load_dataset('json', data_files={"train":"/content/drive/MyDrive/DaoTC2/Finetuning/extractive-qa-mrc/data-bin/unify/train.json", "validation": "/content/drive/MyDrive/DaoTC2/Finetuning/extractive-qa-mrc/data-bin/unify/dev.json", "test": "/content/drive/MyDrive/DaoTC2/Finetuning/extractive-qa-mrc/data-bin/unify/test.json"})
    dataset.filter(assert_sample)
    dataset = dataset.map(format_sample)
    print(dataset)
    
    # all_data = dataset.train_test_split(test_size=0.1)
    train = dataset['train'] 
    valid = dataset['validation']
    test = dataset['test']

    train_set.append(train)
    valid_set.append(valid)
    test_set.append(test)
    
    train_dataset = concatenate_datasets(train_set)
    valid_dataset = concatenate_datasets(valid_set)
    test_dataset = concatenate_datasets(test_set)

    train_dataset.save_to_disk('data-bin/processed/train.dataset')
    valid_dataset.save_to_disk('data-bin/processed/valid.dataset')
    test_dataset.save_to_disk('data-bin/processed/test.dataset')

    print("Train: {} samples".format(len(train_dataset)))
    print("Valid: {} samples".format(len(valid_dataset)))
    print("Valid: {} samples".format(len(test_dataset)))
