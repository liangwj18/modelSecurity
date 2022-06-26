import json

x = json.load(open("question_output.json",'r'))
with open("eva2_question_output.csv",'w') as f:
    for xx in x:
        f.write(xx+"\n")