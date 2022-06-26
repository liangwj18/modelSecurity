# from ..utils import screen_translate_file_name
import sys
sys.path.append("../..")
from utils import screen_translate_file_name
import json
from tqdm import tqdm

new_corpus_dataset = []

for name in ['./train', './test', './dev']:
    with open(name+".txt", 'r') as f:
        lines = f.readlines()
        for i in tqdm(range(0, len(lines), 2)):
            txt = lines[i].split("\t")[1:]
            new_corpus_dataset.append([])
            for sentence in txt:
                new_corpus_dataset[-1].append(sentence.split(' '))

json.dump(new_corpus_dataset, open(screen_translate_file_name, 'w'))

