import json

model_names = [
    'Eva',
    'Eva2',
    'CPM',
    'CDial-GPT',
    'Zhouwenwang'
]
import os
for model_name in model_names:
    path = os.path.join(".",model_name+"-prob_formats_1_gen","prob_formats_11_gen.json")
    js = json.load(open(path,'r'))
    f1 = open(os.path.join(".",model_name+"-prob_formats_1_gen","0.csv"),'w')
    f2 = open(os.path.join(".",model_name+"-prob_formats_1_gen","1.csv"),'w')
    f = [f1,f2]
    for a in js:
        x = a[0].split("问题：")[1].replace("\n回答：",'[SEP]')
        for res in a[1]:
            f[a[2]].write(x+res+"\n")
