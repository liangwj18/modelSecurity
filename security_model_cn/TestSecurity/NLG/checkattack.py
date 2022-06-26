from Analyzer.SentenceAttack_Analyzer import SentenceAttack_Analyzer

analyzer = SentenceAttack_Analyzer("COLD_Detector", "cuda:0")  
models = [
    # 'CDial-GPT',
    # 'Eva',
    # 'Eva2',
    # 'CPM',
    'Zhouwenwang'
]

import os
import json
from tqdm import tqdm

# for model in models:
#     file_path = os.path.join("/home/liangwenjie/security_model_cn/TestSecurity/common_result",model+"-question","question_output.json")
#     js = json.load(open(file_path,'r'))
#     attack = 0
#     for a in tqdm(js):
#         x = a.split("[SEP]")[1]
#         y = analyzer.predict(x)
#         if y == 1:
#             attack+=1
#     f = open("attack.txt",'a')
#     print(model, attack/len(js))
#     f.write(model+ " "+ str(attack/len(js))+"\n")
# EVa  ZHouwenwang
with open("/home/liangwenjie/security_model_cn/TestSecurity/common_result/Zhouwenwang-question/eva2_question_output.csv",'r') as ff:
    lines = ff.readlines()
    attack = 0
    for line in tqdm(lines):
        if len(line.split("[SEP]"))<2:
            continue
        x = line.split("[SEP]")[1].split("\n")[0]
        y = analyzer.predict(x)
        if y == 1:
            attack+=1
    print(attack/len(lines))