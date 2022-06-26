import json
f = json.load(open("question_output.json",'r'))

# json.dump(new_question, open("question_output_new.json",'w'))
with open("round2.csv",'w', encoding = 'utf-8') as ff:
    for k in f:
        ff.write(k+"\n")
