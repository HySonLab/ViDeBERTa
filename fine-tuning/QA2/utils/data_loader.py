import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
from nltk import word_tokenize
# xlm-roberta-base
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base",
                                          cache_dir='./model-bin/cache',
                                          #local_files_only=True
                                         )


def compute_metrics(eval_pred):
    metric = datasets.load_metric("squad", cache_dir='./log/metric')
    print(eval_pred)
    logits, labels = eval_pred
    logits = list(zip(logits[0], logits[1]))
    labels, span_ids, samples_input_ids, word_lengths = list(zip(labels[0], labels[1])), labels[2], labels[3], labels[4]
    predictions = []
    references = []
    for idx, (predict, span_truth, input_ids, sample_words_length) in enumerate(
            list(zip(logits, span_ids, samples_input_ids, word_lengths))):
        span_truth = np.delete(span_truth, np.where(span_truth == -100))
        input_ids = np.delete(input_ids, np.where(input_ids == -100))

        # Get the most likely beginning of answer with the argmax of the score
        answer_start = sum(sample_words_length[:np.argmax(predict[0])])
        # Get the most likely end of answer with the argmax of the score
        answer_end = sum(sample_words_length[:np.argmax(predict[1]) + 1])

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answer_truth = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(span_truth))

        predictions.append({'prediction_text': answer, 'id': str(idx)})
        references.append({'answers': {'answer_start': [answer_start], 'text': [answer_truth]}, 'id': str(idx)})
    print(predictions, references)
    results = metric.compute(predictions=predictions, references=references)
    return results


def data_collator(samples):
    if len(samples) == 0:
        return {}

    # for sample in samples:
    #     start_idx = sum(sample['words_lengths'][:sample['start_idx']])
    #     end_idx = sum(sample['words_lengths'][:sample['end_idx'] + 1])
    #     if len(sample['answer_text']) > 0:
    #         answer_predict = set(
    #             list(' '.join(word_tokenize(tokenizer.decode(sample['input_ids'][start_idx:end_idx])))))
    #         answer_truth = set(list(' '.join(word_tokenize(sample['answer_text']))))
    #         if answer_predict != answer_truth:
    #             print("{}\n{}\n\n-------\n".format(
    #                 ' '.join(word_tokenize(tokenizer.decode(sample['input_ids'][start_idx:end_idx]))),
    #                 ' '.join(word_tokenize(sample['answer_text'])).replace('``', '"').replace("''", '"'),
    #                 sample))

    for sample in samples:
        start_idx = sum(sample['words_lengths'][:sample['start_idx']])
        end_idx = sum(sample['words_lengths'][:sample['end_idx'] + 1])
        sample['span_answer_ids'] = sample['input_ids'][start_idx:end_idx]

    def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    input_ids = collate_tokens([torch.tensor(item['input_ids']) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']) for item in samples], pad_idx=0)
    answer_start = collate_tokens([torch.tensor([item['start_idx']]) for item in samples], pad_idx=0)
    answer_end = collate_tokens([torch.tensor([item['end_idx']]) for item in samples], pad_idx=0)
    span_answer_ids = collate_tokens([torch.tensor(item['span_answer_ids']) for item in samples],
                                     pad_idx=-100)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
        'start_positions': answer_start,
        'end_positions': answer_end,
        'span_answer_ids': span_answer_ids
    }

    return batch_samples


def tokenize_function(example):
    example["question"] = example["question"].split()
    example["context"] = example["context"].split()
    max_len_single_sentence = tokenizer.max_len_single_sentence
    # max_len_single_sentence = 368

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["question"]]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["context"]]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in
            i]) > max_len_single_sentence - 1:
        question_ids = [j for i in question_sub_words_ids for j in i]
        context_ids = [j for i in context_sub_words_ids[:example['answer_word_end_idx'] + 1] for j in i]
        remain_tokens = max_len_single_sentence - 1 - len(question_ids)
        if len(question_ids + context_ids) < max_len_single_sentence - 1:
            context_sub_words_ids_revise = context_sub_words_ids[:example['answer_word_end_idx'] + 1]
            idx = example['answer_word_end_idx'] + 1
            while len([j for i in (context_sub_words_ids_revise + [context_sub_words_ids[idx]]) for j in
                       i]) < remain_tokens and idx < len(context_sub_words_ids):
                context_sub_words_ids_revise.append(context_sub_words_ids[idx])
                idx += 1
            context_sub_words_ids = context_sub_words_ids_revise
        else:
            valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "start_idx": (example['answer_word_start_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "end_idx": (example['answer_word_end_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "valid": valid
    }


def get_dataloader(train_path, valid_path, batch_size=2, num_proc=10):
    train_set = datasets.load_from_disk(train_path)
    valid_set = datasets.load_from_disk(valid_path)
    print("Train set: ", len(train_set))
    print("Valid set: ", len(valid_set))
    # unique_tags = set(tag for doc in tags for tag in train_set)
    train_set = train_set.shuffle().map(tokenize_function, batched=False, num_proc=num_proc).filter(
        lambda example: example['valid'], num_proc=num_proc)
    valid_set = valid_set.map(tokenize_function, batched=False, num_proc=num_proc).filter(
        lambda example: example['valid'], num_proc=num_proc)
    # train_set = train_set.sort('src_ids_len')
    # valid_set = valid_set.sort('src_ids_len')

    print("Train set: ", len(train_set))
    print("Valid set: ", len(valid_set))
    return train_set, valid_set
    #return train_set, valid_set.shard(500, 0)
    #
    # train_dataloader = DataLoader(
    #     train_set, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    # )
    # valid_dataloader = DataLoader(
    #     valid_set, batch_size=batch_size, collate_fn=data_collator
    # )
    #
    # return train_dataloader, valid_dataloader


def build_target_dictionary():
    data_set = datasets.load_from_disk('./data-bin/processed/train.dataset')
    labels = set([item['language'] for item in data_set])
    labels2id = {tag: idx for idx, tag in enumerate(labels)}
    # id2labels = {idx: tag for tag, idx in labels2id.items()}

    tags2id = {
        "same": 1,
        "change": 0
    }
    # id2tags = {idx: tag for tag, idx in tags2id.items()}

    return labels2id, tags2id


if __name__ == "__main__":

    train_dataset, valid_dataset = get_dataloader(
        train_path='./data-bin/processed/train.dataset',
        valid_path='./data-bin/processed/valid.dataset'
    )
    print(train_dataset)
    from tqdm import tqdm

    for batch in tqdm(train_dataset):
        pass
        # print(batch['tgt_words_ids'].shape)

    for batch in tqdm(valid_dataset):
        pass
        # print(batch['tgt_words_ids'].shape)
