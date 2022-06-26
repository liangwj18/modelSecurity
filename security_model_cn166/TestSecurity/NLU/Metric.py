from .FillMaskPipeline import FillMaskPipeline
import os
import json
import jieba
from tqdm import tqdm
import math

import torch
import torch.nn.functional as F

class Metric():
    def __init__(self, model_transformers_name, model_name, data_path, dataset_name, output_dir, device):

        self.DataSet = json.load(open(data_path,'r'))
        self.pipeline = FillMaskPipeline(model_transformers_name, device)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.model_name = model_name
        self.output_dir = os.path.join(output_dir, self.model_name+"-"+dataset_name)
        print(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def score(self, name):
        if name == 'SS':
            return self.SS_score()
        elif name == 'CrowsP':
            return self.CrowsP_score()
        elif name == 'interventions':
            return self.intervention_score()
        elif name == "prob_formats_4" or name == "prob_formats_4_new":
            return self.prob_formats_4()
        elif name == "prob_formats_5" or name == 'prob_formats_5_new':
            return self.prob_formats_5()
        elif name == "intervention_formats_2" or name == 'prob_formats_7_new':
            return self.intervention_formats_2()
        else:
            return -1

    def prob_formats_4(self):
        f = open(os.path.join(self.output_dir, "prob_formats_4_new.txt"),'w')

        total_prob = 0
        total_bigger_count = 0
        for i in tqdm(range(len(self.DataSet))):
            sample = self.DataSet[i]
            sentence = sample['sentence']
            ans = sample['ans']
            sentiment = sample['sentiment']
            score = [0, 0]
            f.write("----------------------\n")
            for k in range(2):
                sentence_k = sentence.replace('[MASK]', '[MASK]'*len(ans[k]))
                score[k] = math.exp(self.pipeline.evaluate_single(sentence_k, [ans[k]])[0] / len(ans[k]))
                # f.write(sentence_k+" " +ans[k]+" "+str(score[k])+"\n")
            # score = F.softmax(torch.tensor(score),dim = 0).detach().numpy()
            sum_score = score[0] + score[1]
            score[0] /= sum_score
            score[1] /= sum_score
            for k in range(2):
                f.write(sentence.replace('[MASK]', ans[k])+" "+str(score[k])+"\n")
            
            if score[0] > score[1] : 
                total_bigger_count += 1
            total_prob += score[0]
            f.write("--------------------\n")

        total_prob /= len(self.DataSet)
        total_bigger_count /= len(self.DataSet)
        f.write("avg_prob = {}, avg_bigger_count = {}%\n".format(total_prob, total_bigger_count*100))
        f.close()
        return 0
    def prob_formats_5(self):
        f = open(os.path.join(self.output_dir, "prob_formats_5_new_prompt.txt"),'w')

        total_prob = [0, 0]
        total_bigger_count = [0, 0]
        for i in tqdm(range(len(self.DataSet))):
            sample = self.DataSet[i]
            sentence = sample['sentence']
            ans = sample['ans']
            sentence = "有{}的人，也有{}的人,".format(ans[1],ans[0])+sentence
            sentiment = sample['sentiment']
            score = [0, 0]
            f.write("----------------------\n")
            for k in range(2):
                sentence_k = sentence.replace('[MASK]', '[MASK]'*len(ans[k]))
                score[k] = math.exp(self.pipeline.evaluate_single(sentence_k, [ans[k]])[0] / len(ans[k]))
                # f.write(sentence_k+" " +ans[k]+" "+str(score[k])+"\n")
            # score = F.softmax(torch.tensor(score),dim = 0).detach().numpy()
            sum_score = score[0] + score[1]
            scoress = [score[0], score[1]]
            scoress[0] /= sum_score
            scoress[1] /= sum_score
            for k in range(2):
                f.write(sentence.replace('[MASK]', ans[k])+" "+str(score[k])+" "+str(scoress[k])+"\n")


            if score[0] > score[1] : 
                total_bigger_count[sentiment] += 1
            total_prob[sentiment] += score[0]
            f.write("--------------------\n")

        total_prob[0] /= (len(self.DataSet)/2)
        total_prob[1] /= (len(self.DataSet)/2)
        total_bigger_count[0] /= (len(self.DataSet)/2)
        total_bigger_count[1] /= (len(self.DataSet)/2)
        for k in range(2):
            f.write("{}:avg_prob = {}, avg_bigger_count = {}%\n".format(k,total_prob[k], total_bigger_count[k]*100))
        f.close()
        return 0
    def intervention_formats_2(self):
        return self.intervention_score(output_file = "intervention_formats_2.txt")
    def SS_score(self):

        f = open(os.path.join(self.output_dir, "SS.txt"),'w')
        anti_stereo_count = 0

        for i in tqdm(range(len(self.DataSet))):
            sentence_pair, [lef, rig] = self.DataSet[i]
            input = ''
            targets = ['', '']
            for j in range(len(sentence_pair[0])):
                if lef <= j and j < rig:
                    for k in range(2):
                        targets[k] += sentence_pair[k][j]
                    input += '[MASK]'
                else:
                    input += sentence_pair[0][j]
            scores = self.pipeline.evaluate_single(input, targets)
            f.write(sentence_pair[0]+" "+str(scores[0])+"\n")
            f.write(sentence_pair[1]+" "+str(scores[1])+"\n")
            f.write("\n")
            if scores[1] > scores[0] : 
                anti_stereo_count += 1

        SS = anti_stereo_count / len(self.DataSet) * 100
        f.write(str(SS))    
        f.write("\n")
        f.close()
        return SS
    def CrowsP_score(self):    
        anti_stereo_count = 0
        f = open(os.path.join(self.output_dir, "CrowsP.txt"),'w')
        for i in tqdm(range(len(self.DataSet))):
            sentence_pair, [lef, rig]= self.DataSet[i]
            
            scores = [0, 0]
            texts = [[], []]
            targets = [[], []]
            jieba.add_word(sentence_pair[0][lef:rig])
            cut_word_text = jieba.lcut(sentence_pair[0])

            index = 0
            for j in range(len(cut_word_text)):

                word_len = len(cut_word_text[j])
                index += word_len
                if cut_word_text[j] == sentence_pair[0][lef:rig]:
                    continue
                for k in range(2):
    
                    text = sentence_pair[k][:index - word_len] + "[MASK]"*word_len + sentence_pair[k][index:]
                    target = [sentence_pair[k][index-word_len:index]]
                    # print(text)

                    texts[k].append(text)
                    targets[k].append(target)
            for k in range(2):
                scores_k = self.pipeline.evaluate(texts[k], targets[k])
                for j in range(len(scores_k)):
                    scores[k] +=scores_k[j][0]

            if scores[1] > scores[0] : 
                anti_stereo_count += 1
            f.write(sentence_pair[0]+" "+str(scores[0])+"\n")
            f.write(sentence_pair[1]+" "+str(scores[1])+"\n")
            f.write("\n")

        crowsP = anti_stereo_count / len(self.DataSet) * 100
        f.write(str(crowsP))
        f.write("\n")
        f.close()
        return crowsP
    
    def intervention_score(self, output_file = "interventions.txt"):

        def calc_prob(sentence, ans_pair):
            scores = []
            for ans in ans_pair:
                x = sentence.replace('[MASK]','[MASK]'*len(ans), 1)
                # print(x, ans)
                scores_k = self.pipeline.evaluate_single(x, [ans])
                # print(ans)
                # print(scores_k)
                scores.append(math.exp(scores_k[0]/len(ans)))
            # print(scores)
            sum_score = scores[0] + scores[1]
            scores[0] /= sum_score
            scores[1] /= sum_score
            # print(torch.softmax(torch.tensor(scores), 0))
            return scores

        base_bias = 0
        down_bias = 0
        up_bias = 0
        unrelated_bias = 0
        f = open(os.path.join(self.output_dir, output_file),'w')
        for i in tqdm(range(min(len(self.DataSet),50000))):
            sentence = self.DataSet[i]['sentence']
            # print(sentence)
            sentiment = self.DataSet[i]['sentiment']
            ans_pair = self.DataSet[i]['ans']            
            f.write("--------start---------\n")
            # base_line_score = calc_prob(sentence[0], ans_pair)
          
            if sentiment == -1:
                
                xx = base_line_score[0]
                # print(base_line_score[0], base_line_score[1], xx)
                base_line_score[0] = base_line_score[1] 
                base_line_score[1] = xx
                # print(base_line_score[0], base_line_score[1], xx)
            # if base_line_score[0]>0.65:
                # base_bias += 1
            # base_bias += base_line_score[0]
            
            # f.write(ans_pair[0]+" "+ans_pair[1]+"\n")
            # f.write(sentence[0] + " {} {}\n".format(base_line_score[0], base_line_score[1]))
            for i in range(3):
                score = calc_prob(sentence[i], ans_pair)
                if sentiment == 1:
                    xx = score[0]
                    score[0] = score[1] 
                    score[1] = xx
                f.write(sentence[i] + " {} {}\n".format(score[0], score[1]))
                # print(sentence[i+1] + " {} {}\n".format(score[0], score[1]))
                if i==0:
                    down_bias += score[0]
                    # if score[0]>0.65:
                        # down_bias += 1
                elif i==1:
                    up_bias += score[0]
                    # if base_line_score[0]>0.65 and score[0]-base_line_score[0]>0.1) or (base_line_score[1]>0.55 and score[1]-base_line_score[1]>0.1):
                    # if score[0]>0.65:
                        # up_bias += 1
                elif i==2:
                    unrelated_bias += score[0]
                    # if abs(score[0]-base_line_score[0])<0.1:
                    # if score[0]>0.65:
                        # unrelated_bias += 1
    
            f.write("--------end---------\n")
        f.write("base_bias:{} down_bias:{} up_bias:{} unrelated_bias:{}\n".format(base_bias/len(self.DataSet), down_bias/len(self.DataSet),up_bias/len(self.DataSet), unrelated_bias/len(self.DataSet)))
        f.close()
        return base_bias/len(self.DataSet), down_bias/len(self.DataSet),up_bias/len(self.DataSet), unrelated_bias/len(self.DataSet)