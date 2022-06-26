import os
import json
import jieba
from tqdm import tqdm
import torch.nn.functional as F
import torch
from .Analyzer.SentenceSentiment_Analyzer import SentenceSentiment_Analyzer
from .Analyzer.SentenceAttack_Analyzer import SentenceAttack_Analyzer
import numpy as np
class NLG_Metric():
    def __init__(self, nlg_model, nlg_model_name, dataset_path, dataset_name, output_dir, device, None_prompt):

        self.DataSet = json.load(open(dataset_path,'r'))
        self.nlg_model = nlg_model
        self.nlg_model_name = nlg_model_name

        self.device = device
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.output_dir = os.path.join(output_dir, self.nlg_model_name+"-"+dataset_name)
        print(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.response_output_path = os.path.join(self.output_dir, 'response.json')
        if dataset_path.find("prob") != -1:
            self.verbalizers = self.DataSet[0]['ans']
            self.None_prompt = None_prompt
            self.prob_None_prompts = []
            self.prob_None_prompts_with_WenTiDaAn = []
            for prompt in self.None_prompt:
                print(prompt)
                self.prob_None_prompts.append(self.nlg_model.forward_words(prompt, self.verbalizers))
                self.prob_None_prompts_with_WenTiDaAn.append(self.nlg_model.forward_words("问题：\n答案："+prompt, self.verbalizers))
            
            self.prob_None_prompts = np.average(np.array(self.prob_None_prompts),axis = 0)
            self.prob_None_prompts_with_WenTiDaAn = np.average(np.array(self.prob_None_prompts_with_WenTiDaAn), axis = 0)

            json.dump(self.prob_None_prompts.tolist(),open("./Prob_cf/{}_p0.json".format(nlg_model_name),'w'))
            json.dump(self.prob_None_prompts_with_WenTiDaAn.tolist(),open("./Prob_cf/{}_p1.json".format(nlg_model_name),'w'))
            

    def score(self, name, args):
        if name == 'OpposeSentiment':
            return self.OpposeSentiment_score(args.SentenceSentiment_Analyzer_name)
        elif name == 'Diversity':
            return self.Diversity_score()
        elif name == 'Politeness':
            return self.Politeness_score(args.SentenceAttack_Analyzer_name)
        elif name == 'Career_and_Family':
            return self.Career_and_Family_score(args.Career_and_Family_word_dir)
        elif name == "prob_formats_1" or name == "prob_formats_1_new" or name == "prob_formats_2_new":
            return self.prob_formats_1()
        elif name == "prob_formats_1_gen":
            return self.prob_formats_1_gen()
        elif name == "prob_formats_1_vec":
            return self.prob_formats_1_vec()
        elif name == "prob_formats_2":
            # return self.prob_formats_2()
            return self.prob_formats_1()
        elif name == "prob_formats_3" or name == "prob_formats_3_new":
            return self.prob_formats_3()
        elif name == 'generate_formats_0':
            return self.generate_formats_0(args)
        elif name == 'generate_formats_1':
            return self.generate_formats_1_and_2(args, 1)
        elif name == 'generate_formats_2':
            return self.generate_formats_1_and_2(args, 2)
        elif name == 'intervention_formats_1' or name == 'prob_formats_6_new':
            return self.intervention_formats_1()
        elif name == "question" or name == "round2_input" or name == "question_not_in_round2_and_3":
            return self.question()
        else:
            return -1
    def write_prob(self, sentence, ans_list, f):
        f.write("----------------------\n")
        total_prob = []
        total_bigger_count = []
        # for ans in ans_list:
        score_oris = self.nlg_model.forward_words(sentence, ans_list)
        # print(score_oris)
        for k,score_ori in enumerate(score_oris):
            score = F.softmax(torch.tensor(score_ori), dim = 0).numpy()
            f.write(sentence+" " +ans_list[k][0]+" "+ans_list[k][1]+" "+str(score[0])+" "+str(score[1])+ "\n")
            f.write(str(score_ori[0])+" "+str(score_ori[1])+ "\n")   
            sumscore = score_ori[0] + score_ori[1]
            f.write(str(score_ori[0]/sumscore)+" "+str(score_ori[1]/sumscore)+ "\n")   
            total_prob.append(score_ori[0]/sumscore)
            total_bigger_count.append(int(bool(score_ori[0]>score_ori[1])))

        return total_prob, total_bigger_count, score_oris
    def prob_formats_1(self):
        for j in range(2):
            f = open(os.path.join(self.output_dir, "prob_formats_{}{}.txt".format(str(j),str(j))),'w')
            ff = open(os.path.join(self.output_dir, "prob_formats_{}{}.json".format(str(j),str(j))),'w')
            total_score_oris = []
            total_prob = [[0,0] for i in range(len(self.DataSet[0]['ans']))]
            total_bigger_count = [[0,0] for i in range(len(self.DataSet[0]['ans']))]
            for i in tqdm(range(len(self.DataSet))):
                sample = self.DataSet[i]
                sentence = sample['sentence']
                ans = sample['ans']
                sentiment = sample['sentiment']
                if j == 1:
                    sentence = "问题：" + sentence + "\n" + "答案："
                probs, bigger_counts,score_oris = self.write_prob(sentence, ans, f)
                total_score_oris.append(score_oris)
                for k in range(len(probs)): 
                    total_prob[k][sentiment] += probs[k]
                    total_bigger_count[k][sentiment] += bigger_counts[k]     
                # for k in range(2):
                #     response = self.nlg_model.forward(sentence)
                #     f.write(response+"\n")
            for k in range(len(total_prob)):
                f.write("verbalizer = [{}, {}]; bias_prob = 0: {}, 1: {}, avg: {}, bias_count = 0: {}, 1: {}, avg: {}\n".format(
                    self.DataSet[0]['ans'][k][0], self.DataSet[0]['ans'][k][1],
                    total_prob[k][0] / len(self.DataSet), total_prob[k][1] / len(self.DataSet), (total_prob[k][0]+total_prob[k][1]) / len(self.DataSet) / 2,
                    total_bigger_count[k][0] / len(self.DataSet), total_bigger_count[k][1] / len(self.DataSet), (total_bigger_count[k][0]+total_bigger_count[k][1]) / len(self.DataSet) / 2,
                ))
    def prob_formats_1_gen(self):
        for j in range(1,2,1):
            f = open(os.path.join(self.output_dir, "prob_formats_{}{}_gen.txt".format(str(j),str(j))),'w')
            ff = open(os.path.join(self.output_dir, "prob_formats_{}{}_gen.json".format(str(j),str(j))),'w')
            xx = []
            for i in tqdm(range(min(len(self.DataSet), 10000))):
                sample = self.DataSet[i]
                sentence = sample['sentence']
                ans = sample['ans']
                sentiment = sample['sentiment']
                if j == 1:
                    sentence = "问题：" + sentence + "\n" + "回答："
                responses = [self.nlg_model.forward(sentence, maxLen = 15) for k in range(10)]
                print("--------------------")
                print(sentence)
                print(responses)
                print("--------------------")
                
                for k in range(10):
                    f.write(sentence+ " " + responses[k]+" "+str(sentiment)+"\n")
                xx.append([sentence, responses, sentiment])
                
                # for k in range(2):
                #     response = self.nlg_model.forward(sentence)
                #     f.write(response+"\n")
         
        # total_prob /= len(self.DataSet)
        # total_bigger_count /= len(self.DataSet)
        # f.write("avg_prob = {}, avg_bigger_count = {}%\n".format(total_prob, total_bigger_count*100))
            f.close()
            json.dump(xx,ff)
        return 0
    def prob_formats_1_vec(self):
        for j in range(1,2,1):
            # f = open(os.path.join(self.output_dir, "prob_formats_{}{}_gen.txt".format(str(j),str(j))),'w')
            f0 = open(os.path.join(self.output_dir, "prob_formats_{}0_vec.json".format(str(j))),'w')
            f1 = open(os.path.join(self.output_dir, "prob_formats_{}1_vec.json".format(str(j))),'w')
            xx = []
            vec0 = []
            vec1 = []
            for i in tqdm(range(min(len(self.DataSet), 200))):
                sample = self.DataSet[i]
                sentence = sample['sentence']
                ans = sample['ans']
                sentiment = sample['sentiment']
                if j == 1:
                    sentence = "问题：" + sentence + "\n" + "回答："
                vec = self.nlg_model.forward_vec(sentence, maxLen = 15)

                print("--------------------")
                print(sentence)
                print(responses)
                print("--------------------")
                if (sentiment==0):
                    v0.append(vec)
                else:
                    v1.append(vec)
                # for k in range(10):
                #     f.write(sentence+ " " + responses[k]+" "+str(sentiment)+"\n")
                # xx.append([sentence, responses, sentiment])
                
                # for k in range(2):
                #     response = self.nlg_model.forward(sentence)
                #     f.write(response+"\n")
         
        # total_prob /= len(self.DataSet)
        # total_bigger_count /= len(self.DataSet)
        # f.write("avg_prob = {}, avg_bigger_count = {}%\n".format(total_prob, total_bigger_count*100))
            # f.close()
            json.dump(vec0,f0)
            json.dump(vec1,f1)
        return 0
    def prob_formats_2(self):
        f = open(os.path.join(self.output_dir, "prob_formats_2.txt"),'w')

        total_prob = 0
        total_bigger_count = 0
        for i in tqdm(range(len(self.DataSet))):
            sample = self.DataSet[i]
            sentence = sample['sentence']
            ans = sample['ans']
            sentiment = sample['sentiment']
            
            # print(ans)
            score = self.write_prob(sentence, ans, f)
            # print(score)
            if score[0] > score[1] : 
                total_bigger_count += 1
            total_prob += score[0]
    
        total_prob /= len(self.DataSet)
        total_bigger_count /= len(self.DataSet)
        f.write("avg_prob = {}, avg_bigger_count = {}%\n".format(total_prob, total_bigger_count*100))
        f.close()
        return 0
    def prob_formats_3(self):
        f = open(os.path.join(self.output_dir, "prob_formats_3_prompt.txt"),'w')

        total_prob = 0
        total_bigger_count = 0
        for i in tqdm(range(min(len(self.DataSet),50000))):
            sample = self.DataSet[i]
            sentence = sample['sentence']
            ans = sample['ans']
            sentiment = sample['sentiment']
            # print(ans)
            ans = [ans]
            sentence = "问题：" + sentence + "\n" + "回答："
            probs, bigger_counts,score_oris = self.write_prob(sentence, ans, f)
            score = score_oris[0]
            # print(score)
            if sentiment == 1:
                score[0], score[1] = score[1], score[0]
            if score[0] > score[1] : 
                total_bigger_count += 1
            total_prob += score[0]
      

        total_prob /= len(self.DataSet)
        total_bigger_count /= len(self.DataSet)
        f.write("avg_prob = {}, avg_bigger_count = {}%\n".format(total_prob, total_bigger_count*100))
        f.close()
        return 0

    def generate_formats_0(self, args):
        topics = self.DataSet['topics']
        for topic_i in range(len(topics)):
            response_list = []
            dataset = self.DataSet['data'][topic_i]
            for i in tqdm(range(len(self.DataSet['data'][topic_i]))):
                sentences = dataset[i]
                # print(sentences[0])
                responses = [self.nlg_model.forward(sentences[k]) for k in range(2)]
                response_list.append([sentences,responses])
            json.dump(response_list, open(os.path.join(self.output_dir,"{}_response.json".format(topics[topic_i])),'w'))
            ret = self.OpposeSentiment_score(args.SentenceSentiment_Analyzer_name, topics[topic_i])
            # assert ret != -1
            ret = self.Politeness_score(args.SentenceAttack_Analyzer_name, topics[topic_i])
            # assert ret != -1
            ret = self.Diversity_score(topics[topic_i])
            # assert ret != -1
            ret = self.Career_and_Family_score(None, topics[topic_i],  self.DataSet['specific_word'][topic_i][0], self.DataSet['specific_word'][topic_i][1])
            # assert ret != -1
        return 0
    def generate_formats_1_and_2(self, args, num):
        
        attack_analyzer = SentenceAttack_Analyzer(args.SentenceAttack_Analyzer_name, self.device)    
        # agreement_analyzer = SentenceAgreement_Analyzer(args.SentenceAgreement_Analyzer_name, self.device)    
        f = open(os.path.join(self.output_dir, "generate_formats_{}.txt".format(num)),'w')
        attack_num = 0
        agreement_num = 0
        for i in tqdm(range(len(self.DataSet))):
            sentence = self.DataSet[i]['sentence']
            response = self.nlg_model.forward(sentence)
            is_attack = attack_analyzer.predict(response)
            # is_agreement = agreement_analyzer.predict(response)
            is_agreement = -1
            f.write(sentence+" "+response+" "+"is_attack = {} is_agreement = {}\n".format(is_attack, is_agreement))
            if is_attack == 1:
                attack_num += 1
            if is_agreement == 1:
                agreement_num += 1
        attack_num /= len(self.DataSet)
        agreement_num /= len(self.DataSet)
        f.write("attack = {} agreement = {}\n".format(attack_num, agreement_num))
        return attack_num, agreement_num
    
    def intervention_formats_1(self):

        base_bias = 0
        down_bias = 0
        up_bias = 0
        unrelated_bias = 0
        f = open(os.path.join(self.output_dir, "intervention_formats_1.txt"),'w')
        for i in tqdm(range(len(self.DataSet))):
            sentence = self.DataSet[i]['sentence']
            # print(sentence)
            sentiment = self.DataSet[i]['sentiment']
            ans_pair = self.DataSet[i]['ans']            
            f.write("--------start---------\n")
            # base_line_score = self.nlg_model.forward_words(sentence[0], ans_pair)
          
            # if sentiment == -1:
                
            #     xx = base_line_score[0]
            #     # print(base_line_score[0], base_line_score[1], xx)
            #     base_line_score[0] = base_line_score[1] 
            #     base_line_score[1] = xx
            #     # print(base_line_score[0], base_line_score[1], xx)
            # # if base_line_score[0]>0.65:
            #     # base_bias += 1
            # base_bias += base_line_score[0]
            
            f.write(ans_pair[0]+" "+ans_pair[1]+"\n")
            # f.write(sentence[0] + " {} {}\n".format(base_line_score[0], base_line_score[1]))
            for i in range(3):
                score = self.nlg_model.forward_words(sentence[i], [ans_pair])[0]
                if sentiment == -1:
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
        f.write(" down_bias:{} up_bias:{} unrelated_bias:{}\n".format(down_bias/len(self.DataSet),up_bias/len(self.DataSet), unrelated_bias/len(self.DataSet)))
        f.close()
        return down_bias/len(self.DataSet),up_bias/len(self.DataSet), unrelated_bias/len(self.DataSet)
    def question(self):

        response_list = []
        for i in tqdm(range(len(self.DataSet))):
            sentence = self.DataSet[i]['sentence']
            # sentence2 = self.DataSet[i+1]['sentence']

            sentences = [sentence, sentence, sentence, sentence]
            responses = [self.nlg_model.forward(sentence) for k in range(4)]
            for k in range(4):
                response_list.append(sentences[k]+"[SEP]"+responses[k][:40])
                # print(response_list[-1])
            # for k in range(4):
            #     response_list.append(sentence+"[SEP]"+self.nlg_model.forward(sentence)[:40])
                # print(response_list[-1])
            # response_list.append(sentence+"[SEP]"+self.nlg_model.forward(sentence)[:40])
            if i % 200 == 0:
                json.dump(response_list, open(os.path.join(self.output_dir, "question_output.json"),'w'))
                # json.dump(response_list, open(os.path.join(self.output_dir, "question_output.json"),'w'))
        return 0

    def Generate_Response(self, args):    
        if (not args.reGenerateResponse) and os.path.exists(self.response_output_path):
            return 
        response_list = []
        if args.debug:
            maxLen = min(200, len(self.DataSet))
        else:
            maxLen = len(self.DataSet)
        for i in tqdm(range(maxLen)):
            sentences = self.DataSet[i]
            responses = [self.nlg_model.forward(sentences[i]) for i in range(2)]
            response_list.append(responses)
        json.dump(response_list, open(self.response_output_path, 'w'))

    def OpposeSentiment_score(self, SentenceSentiment_Analyzer_name, topic_name = ""):
        analyzer = SentenceSentiment_Analyzer(SentenceSentiment_Analyzer_name, self.device)
        # 其他model
        # 情感词典 像vader 
        # 自己训练一个 情感分析模型 
        # 百度API
        # print(analyzer.polarity_scores)
        if len(topic_name) == 0:
            f = open(os.path.join(self.output_dir, "OpposeSentiment.txt"),'w')
        else:
            f = open(os.path.join(self.output_dir, "{}_OpposeSentiment.txt".format(topic_name)),'w')
        # 同意的词 否定的词
        # 专业 计算机系 化学 生物  艺术 普通高考
        # profession 不包含乞丐这种   护士医生律师程序员   医生（男） 护士（女）
        # 地域 城乡 
        # 
        # 筛选数据 困惑度
        # 用google 生成数据集
        # 整理数据
        Positive = [0, 0]
        Negative = [0, 0]
        if len(topic_name) == 0:
            response_list = json.load(open(self.response_output_path, 'r'))
        else:
            response_list = json.load(open(os.path.join(self.output_dir,"{}_response.json".format(topic_name)), 'r'))
        for i in tqdm(range(len(response_list))):
            if len(topic_name) == 0:
                sentences = self.DataSet[i]
                responses = response_list[i]
            else:
                sentences, responses = response_list[i]
            # print(sentences, responses)
            vs = [analyzer.predict(responses[i]) for i in range(2)]
            # print(vs)
            for k in range(2):
                if vs[k] == 1:
                    Positive[k] += 1
                if vs[k] == -1:
                    Negative[k] += 1
            f.write("-"*20)
            f.write("\n")
            # print(sentences)
            for k in range(2):
                f.write("Sentences: "+"".join(sentences[k]))
                f.write("\n")
                f.write("Responses: "+responses[k])
                f.write("\n")
                f.write("Scores:    "+str(vs[k]))
                f.write("\n")
            f.write("-"*20)
            f.write("\n")

        for i in range(2):
            Positive[i] /= len(response_list)
            Negative[i] /= len(response_list)
    
        f.write("Positive: {} {}\n".format(Positive[0], Positive[1]))
        f.write("Negative: {} {}\n".format(Negative[0], Negative[1]))
        f.close()
        return Positive, Negative
    
    def Politeness_score(self, SentenceAttack_Analyzer_name, topic_name = ""):
        analyzer = SentenceAttack_Analyzer(SentenceAttack_Analyzer_name, self.device)    
        if len(topic_name) == 0:
            f = open(os.path.join(self.output_dir, "Attack.txt"),'w')
        else:
            f = open(os.path.join(self.output_dir, "{}_Attack.txt".format(topic_name)),'w')
        Attack = [0, 0]
        nonAttack = [0, 0]
        if len(topic_name) == 0:
            response_list = json.load(open(self.response_output_path, 'r'))
        else:
            response_list = json.load(open(os.path.join(self.output_dir,"{}_response.json".format(topic_name)), 'r'))
        for i in tqdm(range(len(response_list))):
            if len(topic_name) == 0:
                sentences = self.DataSet[i]
                responses = response_list[i]
            else:
                sentences, responses = response_list[i]
            vs = [analyzer.predict(responses[i]) for i in range(2)]
            for k in range(2):
                if vs[k] == 1:
                    Attack[k] += 1
                if vs[k] == -1:
                    nonAttack[k] += 1
            f.write("-"*20)
            f.write("\n")
            for k in range(2):
                f.write("Sentences: "+"".join(sentences[k]))
                f.write("\n")
                f.write("Responses: "+responses[k])
                f.write("\n")
                f.write("Scores:    "+str(vs[k]))
                f.write("\n")
            f.write("-"*20)
            f.write("\n")

        for i in range(2):
            Attack[i] /= len(response_list)
            nonAttack[i] /= len(response_list)
    
        f.write("Attack: {} {}\n".format(Attack[0], Attack[1]))
        f.write("nonAttack: {} {}\n".format(nonAttack[0], nonAttack[1]))
        f.close()
        return Attack, nonAttack
    
    def Diversity_score(self, topic_name = ""):
        
        if len(topic_name) == 0:
            response_list = json.load(open(self.response_output_path, 'r'))
            f = open(os.path.join(self.output_dir, "Diversity.txt"),'w')
        else:
            response_list = json.load(open(os.path.join(self.output_dir,"{}_response.json".format(topic_name)), 'r'))
            f = open(os.path.join(self.output_dir, "{}_Diversity.txt".format(topic_name)),'w')
        
        unigram_dic = [{},{}]
        bigram_dic = [{},{}]
        for i in tqdm(range(len(response_list))):
            _,responses = response_list[i]
            for k in range(2):
                for character in responses[k]:
                    unigram_dic[k][character] = 1
                seg = jieba.lcut(responses[k])
                for word in seg:
                    bigram_dic[k][word] = 1
        
        f.write("Unigram_Num:{} {}\n".format(len(unigram_dic[0]), len(unigram_dic[1])))
        f.write("Bigram_Num:{} {}\n".format(len(bigram_dic[0]), len(bigram_dic[1])))
        f.close()
        diversity_score = [(len(unigram_dic[k]) + len(bigram_dic[k])) / 2.0 for k in range(2)]
        return diversity_score

    def Career_and_Family_score(self, word_dic_dir, topic_name = "", good_dic = None, bad_dic = None):
        if len(topic_name) == 0:
            response_list = json.load(open(self.response_output_path, 'r'))
            f = open(os.path.join(self.output_dir, "Career_and_Family.txt"),'w')
            career_word_dic = json.load(open(os.path.join(word_dic_dir, 'career.json'),'r'))
            family_word_dic = json.load(open(os.path.join(word_dic_dir, 'family.json'),'r'))

        else:
            response_list = json.load(open(os.path.join(self.output_dir,"{}_response.json".format(topic_name)), 'r'))
            f = open(os.path.join(self.output_dir, "{}_Specific.txt".format(topic_name)),'w')
            career_word_dic = good_dic
            family_word_dic = bad_dic


        career_word = [0, 0]
        family_word = [0, 0]
        for i in tqdm(range(len(response_list))):
            _,responses = response_list[i]
            for k in range(2):
               
                seg = jieba.lcut(responses[k])
                for word in seg:
                    if word in career_word_dic:
                        career_word[k] += 1
                    elif word in family_word_dic:
                        family_word[k] += 1
        
        f.write("Career_Num:{} {}\n".format(career_word[0], career_word[1]))
        f.write("Family_Num:{} {}\n".format(family_word[0], family_word[1]))
        f.close()
        return career_word, family_word
