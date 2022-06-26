from abc import ABCMeta, abstractmethod
import os
import json

import math
import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead, BertTokenizerFast

class GenerateDataBase():
    def __init__(self, args, folder_name, file_name, debug = False):
        if not os.path.exists("./"+folder_name):
            os.mkdir("./"+folder_name)
        self.output_path = os.path.join(".",folder_name, file_name+".json")
        self.sentences = []
        self.debug = debug
        self.device = args.device
                
        self.cpm_filter_tokenizer = BertTokenizerFast.from_pretrained("ckiplab/gpt2-base-chinese")
        self.cpm_filter_model = AutoModelWithLMHead.from_pretrained("ckiplab/gpt2-base-chinese")
        self.cpm_filter_model.to(args.device)

    def add(self, sentence):
        self.sentences.append(sentence)
    
    def filter_perplexity(self, sentences):
        perplexitys = []
        filter_sentences = []
        with torch.no_grad():
            for sentence in tqdm(sentences):
                # sentence = "客观来说，大学生在吹捧方面比中专生有优势"
                tokens = self.cpm_filter_tokenizer.tokenize(sentence)
                input_ids = self.cpm_filter_tokenizer.convert_tokens_to_ids(tokens)
                input_tensor = torch.tensor(input_ids).to(self.device)
                outputs = self.cpm_filter_model(input_tensor, labels = input_tensor.clone())
                loss = outputs[0]
                perplexity = loss.item()
                if perplexity > 8.5:
                    filter_sentences.append(sentence)
                    perplexitys.append(perplexity)
        return filter_sentences, perplexitys

    def _generate(self, func):
        outputs = []
        for sentence in self.sentences:
            sen_outputs = func(sentence)
            for sen_output in sen_outputs:
                outputs.append(sen_output)
                if self.debug:
                    print(sen_output)
        outputs = self.filter_perplexity(outputs)
        print("all_size : {}".format(len(outputs)))
        json.dump(outputs, open(self.output_path,'w', encoding = 'gbk'))

    def generate(self):
        self._generate(lambda x:x.generate())
    
    def generate_CrowsP(self):
        self._generate(lambda x:x.generate_CrowsP_format())

    def generate_SS(self):
        self._generate(lambda x:x.generate_SS_format())