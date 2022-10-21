# import datasets

# raw_datasets = datasets.load_dataset('text', data_files="../dataset/vi_2e8.txt")
# print(raw_datasets)
# print(raw_datasets['train'][:10])
# def split_data(examples):
#         tokens = []
#         labels = []
#         for i in examples['text']:
#             if(len(i)>0): 
#                 _split = i.split(" ")
#                 # print(_split)
#                 tokens.append(_split[0])
#                 labels.append(_split[1])
        
#         examples['text'] = tokens
#         examples['labels'] = labels
#         return examples

# raw_datasets = raw_datasets.map(split_data, batched=True)
# print(raw_datasets['train'][:10])
# from multiprocessing import Pool
# from pyvi import ViTokenizer, ViPosTagger, ViUtils
# import time

# def segment(line):
#     return ViTokenizer.tokenize(line)

# output = segment(u"Xin chÃ o anh HÃ¹ng, hÃ´m nay lÃ  ngÃ y thá»© Báº£y. ChÃºng tÃ´i chuáº©n bá»‹ tham gia cuá»™c há»p vá»›i cÃ¡c anh mentors.")
# print(output)
# pool = Pool(16)
# start_time = time.perf_counter()
# fo = open("../dataset/vi_cc100_1e8_segment.txt", "w")
# with open("/media/data/huypn10/Vietnamese%20Pretrained%20Model/pre-training/dataset/cc100_1e7.txt", "r") as source_file:
#     results = pool.map(segment, source_file, 100)
#     content = ''.join(results)
#     fo.write(content)
# fo.close()
# finish_time = time.perf_counter()
# print(f"Finised in {finish_time-start_time} seconds")

# from transformers import AutoTokenizer
# from pyvi import ViTokenizer
# tokenizer = AutoTokenizer.from_pretrained('../tokenizer/spm')
# sentence = "Tiếng Việt, cũng gọi là tiếng Việt Nam[9] hay Việt ngữ là ngôn ngữ của người Việt và là ngôn ngữ chính thức tại Việt Nam. Đây là tiếng mẹ đẻ của khoảng 85% dân cư Việt Nam cùng với hơn 4 triệu người Việt kiều. Tiếng Việt còn là ngôn ngữ thứ hai của các dân tộc thiểu số tại Việt Nam và là ngôn ngữ dân tộc thiểu số được công nhận tại Cộng hòa Séc."
# print(tokenizer.tokenize(ViTokenizer.tokenize(sentence)))

# from huggingface_hub import HfApi
# api = HfApi()

# api.upload_file(path_or_fileobj="../tokenizer/spm/tokenizer.json", 
#                 path_in_repo="tokenizer.json", 
#                 repo_id="Aehus/vi-DeBERTa-v3-xsmall",
# )
# with open("../dataset/vi_2e8.txt","r") as f:
#     cnt = 0
#     for line in f:
#         cnt += 1
#         if(cnt<10):
#             print(segment(line))
#     print(cnt)
# import lzma
# with open('../dataset/vi_2e8.txt','r') as f:
#     cnt = 0
#     for i in f:
#         cnt += 1
#         if cnt<10: 
#             print(i, end = " ")
#             print(len(i))
#         else: break
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("../tokenizer/spm")

# print(tokenizer.part_of_whole_word('▁trong'))
# print(tokenizer.convert_tokens_to_ids('▁trong'))
import datasets
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerFast,DebertaV2TokenizerFast
from tokenizers import SentencePieceBPETokenizer
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Tokenizer name')
    parser.add_argument('--source_dir', type=str, default=None, help='Source file')
    parser.add_argument('--destination_dir', type=str, default=None, help='Cache dir')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache dir')
    return parser.parse_args()
if __name__ == '__main__':
    args = args_parser()
    #
    # # Build a tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')
    print(tokenizer.vocab[:10])
    # bpe_tokenizer.save_pretrained('../tokenizer/tmp')
        