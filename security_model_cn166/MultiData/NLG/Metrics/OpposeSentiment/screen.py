import os
import json

from utils import screen_translate_file_name
import argparse
from tqdm import tqdm
import jieba

def check_words_in_str_num(st, word_pairs):
    word_list = [word_pairs[i][j] for i in range(len(word_pairs)) for j in range(2)]
    cnt = 0
    for word in word_list:
        idx = st.find(word)
        if idx > -1 :
            cnt += 1
            if st[idx + len(word):].find(word) > -1:
                cnt += 1
            if cnt >= 2:
                return cnt
    return cnt
    

def main():
    Gender_words = [['他', '她'], ['男', '女']]
    Region_words = []
    Profession_words = []
    Education_words = [['大学生', '中专生'], ['本科生', '大专生']]

    Topic_Dic = {
        'Gender': Gender_words,
        'Region': Region_words,
        'Profession': Profession_words,
        'Education': Education_words
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", choices = ['Gender','Region','Profession','Education'])
    parser.add_argument("--data_path", default = '../../Datasets')
    parser.add_argument("--output_path", default = "../../")
    parser.add_argument("--output_name", default = "OpposeSentiment.json")
    args = parser.parse_args()

    pass_dataset_name = [
        'DoubanConversaionCorpus'
    ]

    new_corpus_dataset = []
    path = args.data_path
    for root, dirs, files in os.walk(path):
        for name in dirs: #各大对话数据集
            corpus_dataset_path = os.path.join(root, name)
            corpus_dataset_translate_path = os.path.join(corpus_dataset_path, screen_translate_file_name)
            if not os.path.exists(corpus_dataset_translate_path):
                print("{} has not been translate format !!".format(corpus_dataset_path))
                continue
            if name in pass_dataset_name:
                print("PASS...  {}".format(corpus_dataset_path))
                continue
            print("START...  {}".format(corpus_dataset_path))
            corpus_dataset = json.load(open(corpus_dataset_translate_path, 'r'))
            for context_i, context in tqdm(enumerate(corpus_dataset)):
                for sentence in context:
                    cnt = check_words_in_str_num("".join(sentence), Topic_Dic[args.topic])
                    if cnt != 1:
                        continue
                    for word_pair in Topic_Dic[args.topic]:
                        for k in [0, 1]:
                            if (word_pair[k] in sentence) and (word_pair[k^1] not in sentence):
                                idx = sentence.index(word_pair[k])
                                if word_pair[k] in sentence[idx+1:]:
                                    continue
                                new_sentence_pair = [
                                    sentence[:idx] + [word_pair[i]] + sentence[idx+1:]
                                    for i in [0, 1]
                                ]
                                new_corpus_dataset.append(new_sentence_pair)
                                break

    if not os.path.exists(os.path.join(args.output_path, args.topic)):
        os.mkdir(os.path.join(args.output_path, args.topic))
    json.dump(new_corpus_dataset, open(os.path.join(args.output_path, args.topic, args.output_name), 'w'))
    for i in range(min(10, len(new_corpus_dataset))):
        print(new_corpus_dataset[i])
if __name__ == "__main__":
    main()