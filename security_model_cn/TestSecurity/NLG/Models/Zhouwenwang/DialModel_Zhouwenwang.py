from ..DialModel_base import DialModelBase
from transformers import AutoTokenizer,BertTokenizer
import torch
import numpy as np
import torch
import paddle
from .fengshen import RoFormerModel
class DialModel_Zhouwenwang(DialModelBase):
    def __init__(self, device):
        self.device = device
    #    
        # sentence = '清华大学位于'
        self.max_length = 40

        self.tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-1.3B")
        self.model = RoFormerModel.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-1.3B").to(device)

    # #get the tokenizer
    def test(self):
        vocab = self.model.embeddings.word_embeddings
        
        sa = self.tokenizer("赞成")
        sb = self.tokenizer("同意")
        sc = self.tokenizer("反对")

        a = self.model.embeddings.forward(torch.tensor([sa['input_ids']]).to(self.device),torch.tensor([sa['token_type_ids']]).to(self.device))
        b = self.model.embeddings.forward(torch.tensor([sb['input_ids']]).to(self.device),torch.tensor([sb['token_type_ids']]).to(self.device))
        c = self.model.embeddings.forward(torch.tensor([sc['input_ids']]).to(self.device),torch.tensor([sc['token_type_ids']]).to(self.device))
        a = a.squeeze(dim = 0)
        b = b.squeeze(dim = 0)
        c = c.squeeze(dim = 0)
        print(a[1:-1].shape)
        print(torch.mean(a[1:-1],axis=0))
        print(torch.mean(a[1:-1],axis=0).shape)
        print(torch.nn.functional.cosine_similarity(torch.mean(a[1:-1], axis = 0),torch.mean(b[1:-1], axis = 0),axis=0))
        print(torch.nn.functional.cosine_similarity(torch.mean(b[1:-1], axis = 0),torch.mean(c[1:-1], axis = 0),axis=0))
        # w2v = self.model.embeddings.forward(self.tokenizer("建筑物"))
        
    def forward(self, input, maxLen = -1):
        input = "".join(input)
        if maxLen == -1:
            maxLen = self.max_length
        self.model.eval()
        sentence = input
        with torch.no_grad():
            encode = [self.tokenizer.cls_token_id]+self.tokenizer.encode(sentence, add_special_tokens=False)
            past_key_values = None
            for i in range(maxLen):
                
                input_ids=torch.tensor([encode]).long().to(self.device)
                token_type_ids=torch.tensor([[1]*len(encode)]).long().to(self.device)
                outputs = self.model(input_ids=input_ids, past_key_values = past_key_values,
                   token_type_ids=token_type_ids,return_dict = True)
                logits = outputs.last_hidden_state
                past_key_values = outputs.past_key_values
                # print(logits.shape)
                # assert 1 == 2
                # past_key_values = outputs['past_key_values']
                logits = torch.nn.functional.linear(
                    logits, self.model.embeddings.word_embeddings.weight)
                logits = torch.nn.functional.softmax(
                    logits, dim=-1).cpu().detach().numpy()[0]
                x = int(np.random.choice(logits.shape[1], p=logits[-1]))
                sentence = sentence + \
                    self.tokenizer.decode(x)
                encode.append(x)
                if sentence[-1] == '。':
                    break
            # print(sentence)
            print("input",input)
            print("sentence",sentence)
            print("response",sentence[len(input):])
            return sentence[len(input):]
        

    def forward_words(self, input, words:list):
        input = "".join(input)
        self.model.eval()
        sentence = input
        with torch.no_grad():
            ret = []
            for word_pair in words:
                xx = []
                for word in word_pair:
                    sentence = input
                    word_ids = self.tokenizer.encode(word, add_special_tokens = False)
                    p = 1
                    encode = [self.tokenizer.cls_token_id]+self.tokenizer.encode(sentence, add_special_tokens=False)
                    for i in range(len(word_ids)):
                        
                        input_ids=torch.tensor([encode]).long().to(self.device)
                        token_type_ids=torch.tensor([[1]*len(encode)]).long().to(self.device)
                        logits = self.model(input_ids=input_ids, 
                        token_type_ids=token_type_ids)[0]
                        logits = torch.nn.functional.linear(
                            logits, self.model.embeddings.word_embeddings.weight)
                        logits = torch.nn.functional.softmax(
                            logits, dim=-1).cpu().detach().numpy()[0]
                   
                        # print(word_ids[i])
                        # print(logits[-1].shape)
                        p*= logits[-1][word_ids[i]]
                        sentence = sentence + word[i]
                        # print(sentence)
                        encode.append(word_ids[i])
                        if sentence[-1] == '。':
                            break
                    p = pow(p, 1/len(word_ids))
                    xx.append(p)
                # print(word_pair, xx)
                ret.append(xx)
            return ret


if __name__ == "__main__":
    model = DialModel_Eva()
    print(model.args)



