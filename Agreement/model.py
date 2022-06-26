from transformers import pipeline
from transformers import BertTokenizer, BertModel,BertForMaskedLM # BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
class AgreementModel(nn.Module):

    def __init__(self, labels_num, dropout, model_with_sentiment, roberta, args):
        super(AgreementModel, self).__init__()
        # if roberta:
        # if args.base_model!="bert-base-chinese":
            
        self.bert =BertModel.from_pretrained(args.base_model) 
            # self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
            # self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert = BertModel.from_pretrained('bert-base-chinese') 
        # print(self.bert)
 
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.model_with_sentiment = model_with_sentiment
        if model_with_sentiment:
            # self.fc_sentiment = nn.Linear(768, 3)
            self.fc_agreement_embedding = nn.Linear(768, labels_num)
            self.fc_agreement = nn.Linear(5+labels_num, labels_num)
        else:
            self.fc = nn.Linear(768, labels_num)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, sentiment = None):
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # context = x[0]  # 输入的句子
        # mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        _, pooled = self.bert(x['input_ids'], attention_mask=x['attention_mask'], return_dict=False)
        # out = self.bert(**x)['logits']
        
        out = self.dropout(pooled)
        if self.model_with_sentiment:
            # sentiment = self.fc_sentiment(pooled)
            # sentiment = self.dropout(sentiment)
            # sentiment = F.softmax(sentiment, dim = 1)

            out = self.fc_agreement_embedding(pooled)
            out = self.dropout(out)
            out = F.softmax(out, dim = 1)
            # print(torch.concat([out, sentiment], dim = 1).shape)
            out = self.fc_agreement(torch.concat([out, sentiment], dim = 1))
            out = self.dropout(out)
            out = F.softmax(out, dim = 1)
            return out
        else:
            out = self.fc(pooled)
            out = self.dropout(out)
            out = F.softmax(out, dim = 1)
            return out