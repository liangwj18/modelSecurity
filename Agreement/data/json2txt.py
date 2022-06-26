import json

file_name = "CDial_GPT"
with open(file_name+".csv",'w') as f:
    ls = json.load(open(file_name+".json",'r'))
    for a in ls:
        f.write(a+"\n")