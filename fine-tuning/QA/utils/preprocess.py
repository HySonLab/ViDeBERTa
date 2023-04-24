import json
from glob import glob
from nltk.classify.decisiontree import f
from tqdm import tqdm
import re
from nltk import word_tokenize as lib_tokenizer

dict_map = dict({})


def word_tokenize(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm

def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def handle_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_read:
        # with open('../data-bin/squad/raw/train-v2.0.json', 'r', encoding='utf-8') as file_read:
        json_data = json.load(file_read)
    qa_data = json_data['data']
    norm_samples = []
    idx = 0
    for item in tqdm(qa_data, total=len(qa_data), desc="Chunk {}".format(file_path)):
        for par in item['paragraphs']:
            context_raw = par['context']
            for qa_sample in par['qas']:
                question = qa_sample['question']
                if len(qa_sample['answers']) > 0:
                    answer = [ans['text'] for ans in qa_sample['answers']]
                    answer_index_raw = [ans['answer_start'] for ans in qa_sample['answers']]
                    # if not qa_sample['is_impossible']:
          
                
                    # answer_raw = qa_sample['answers'][0]['text']
                    # answer_index_raw = qa_sample['answers'][0]['answer_start']
                        # if check == False and context_raw[answer_index_raw: answer_index_raw + len(answer_raw)] == answer_raw:
                        #     check = True
                        #     context_prev = strip_context(context_raw[:answer_index_raw])
                        #     answer = strip_answer_string(answer_raw)
                        #     context_next = strip_context(context_raw[answer_index_raw + len(answer):])
                        #     context_prev = ' '.join(word_tokenize(context_prev))
                        #     context_next = ' '.join(word_tokenize(context_next))
                        
                    # answer = ' '.join(word_tokenize(answer))
                    # answer_start = len("{} {}".format(context_prev, answer).strip()) - len(answer)
                    # question = ' '.join(word_tokenize(question))
                    # context = "{} {} {}".format(context_prev, answer, context_next).strip()

                    norm_samples.append({
                        "id": idx,
                        "context": context_raw,
                        "question": question,
                        "answers": {"answer_start": answer_index_raw, "text": answer}
                        # "answer_text": answer,
                        # "answer_start_idx": len("{} {}".format(context_prev, answer).strip()) - len(answer)
                    })
                    idx = idx+1
                else:
                    context_raw = ' '.join(word_tokenize(context_raw))
                    question = ' '.join(word_tokenize(question))
                    norm_samples.append({
                        "id": idx,
                        "context": context_raw,
                        "question": question,
                        "answers": {"answer_start": [0], "text": ['']}
                    })
                    idx=idx+1
    print(len(norm_samples))
    return norm_samples


if __name__ == "__main__":
    # list_data_file = glob('/content/drive/MyDrive/DaoTC2/Finetuning/QA/extractive-qa-mrc/data-bin/raw/squad/*')
    # dict_data_squad = []
    # for file_name in list_data_file:
    #     dict_data_squad.extend(handle_file(file_name))

    dict_data_squad_train = handle_file("/content/drive/MyDrive/aFPT_RE/Finetuning/QA/extractive-qa-mrc/data-bin/train.json")
    dict_data_squad_dev = handle_file("/content/drive/MyDrive/aFPT_RE/Finetuning/QA/extractive-qa-mrc/data-bin/dev.json")
    dict_data_squad_test = handle_file("/content/drive/MyDrive/aFPT_RE/Finetuning/QA/extractive-qa-mrc/data-bin/test.json")
    # print(dict_data_squad_dev)

    with open('/content/drive/MyDrive/aFPT_RE/Finetuning/extractive-qa-mrc/data-bin/unify1/squad_mrc_train.json', 'w', encoding='utf-8') as file:
      file.write("{}\n".format(json.dumps(dict_data_squad_train, ensure_ascii=False)))
    print("Total: {} samples".format(len(dict_data_squad_train)))
    with open('/content/drive/MyDrive/aFPT_RE/Finetuning/extractive-qa-mrc/data-bin/unify1/squad_mrc_dev.json', 'w', encoding='utf-8') as file:
      file.write("{}\n".format(json.dumps(dict_data_squad_dev, ensure_ascii=False)))
    print("Total: {} samples".format(len(dict_data_squad_dev)))
    with open('/content/drive/MyDrive/aFPT_RE/Finetuning/extractive-qa-mrc/data-bin/unify1/squad_mrc_test.json', 'w', encoding='utf-8') as file:
      file.write("{}\n".format(json.dumps(dict_data_squad_test, ensure_ascii=False)))
    print("Total: {} samples".format(len(dict_data_squad_test)))
