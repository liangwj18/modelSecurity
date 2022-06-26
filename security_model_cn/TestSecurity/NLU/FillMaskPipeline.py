from transformers import BertTokenizer, BertForMaskedLM
# from transformers import XLNetTokenizer, XLNetModel
import torch
import json
import math

class FillMaskPipeline():
    def __init__(self, model_name, device):

        # if pretrainedModel_name == 'Bert':
        #     Tokenizer = BertTokenizer
        #     model = BertForMaskedLM
        # elif pretrainedModel_name == 'xlnet':


        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(device)
        self.device = device

    def get_mask_indexs(self, input_ids):
        mask_indexs = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=True)
        return mask_indexs
    
    def calc_score(self, probs, target):
        total_score = 0
        token = self.tokenizer(target, return_tensors="pt")['input_ids'].tolist()[0][1:-1]
        for i in range(probs.shape[0]):
            prob = probs[i][token[i]]
            total_score += math.log(prob)
        return total_score
    def evaluate_single(self, input, targets, debug = False):
        score_batch = self.evaluate([input], [targets], debug)
        return score_batch[0]
    def evaluate(self, inputs, targets, debug:bool=False):
        
        input_tokens = self.tokenizer(inputs, return_tensors="pt").to(self.device)
        score_batch = []
        with torch.no_grad():
            outputs = self.model(**input_tokens)['logits']
        for batch_i in range(len(inputs)):
            mask_indexs = self.get_mask_indexs(input_tokens['input_ids'][batch_i])
            assert len(mask_indexs) > 0
            logits = outputs[batch_i, mask_indexs[0][0]:mask_indexs[-1][0]+1,:]
            probs = logits.softmax(dim = 1)
            scores = []
            # print(targets)
            for target in targets[batch_i]:
                score = self.calc_score(probs, target)
                scores.append(score)
                # print(target, targets, score, scores)
        
            score_batch.append(scores)
        return score_batch
        


if __name__ == "__main__":
    path = "../MultiData/education/Education/CrowsP.json"
    CrowsP_Dataset = json.load(open(path, 'r'))
    
    pipeline = FillMaskPipeline(
        model_name = 'bert-base-chinese',
        device = 'cuda:0'
    )
    sentence_pair, subject_range = CrowsP_Dataset[0]

    lef, rig = subject_range
    scores = [0, 0]
    texts = [[], []]
    targets = [[], []]

    for i in range(len(sentence_pair[0])):
        if lef <= i and i <= rig:
            continue

        for k in range(2):
            text = sentence_pair[k][:i] + "[MASK]" + sentence_pair[k][i+1:]
            target = [sentence_pair[k][i]]

            texts[k].append(text)
            targets[k].append(target)
    for k in range(2):
        scores_k = pipeline.evaluate(texts[k], targets[k])
        for j in range(len(scores_k)):
            scores[k] += scores_k[j][0]

    if scores[1] > scores[0] : 
        anti_stereo_count += 1
