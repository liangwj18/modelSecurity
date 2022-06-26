import json
import os

def translate(path,name):
    subj_path = os.path.join(path,name+".txt")
    if os.path.exists(subj_path):
        f = open(subj_path, 'r')
        state = 0
        subj_json = {"pair":[],"group":[[],[]]}
        for line in f.readlines():
            line = line.split("\n")[0]
            if len(line)==0:
                break
            if line[0] == '-':
                state += 1
            elif state == 0:
                subj_json['pair'].append(line.split(" "))
            else: subj_json['group'][state-1].append(line)
        json.dump(subj_json, open(os.path.join(path, name+".json"),'w'))

topics = ['Education', 'Gender', 'Profession', 'Religion']
for topic in topics:
    
    translate(os.path.join("Topic",topic), 'subj')
    translate(os.path.join("Topic",topic), 'adj_like_subj')
    
translate(".","adj")
translate(".","verb")
translate(".","context")