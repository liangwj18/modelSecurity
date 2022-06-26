import json
f = json.load(open("question_output.json",'r'))
print(len(f))
new_question = []
for i in range(0,len(f),4):
    question = f[i][0]
    answer = [f[i+j][1] for j in range(4)]
    answer = sorted(answer)
    new_answer = []
    for k in range(4):
        if k == 0 or answer[k]!=answer[k-1]:
            new_answer.append(answer[k])
    new_question.append([question, new_answer])

json.dump(new_question, open("question_output_new.json",'w'))
with open("round1.csv",'w', encoding = 'utf-8') as ff:
    for k in range(len(new_question)):
        for answer in new_question[k][1]:   
            ff.write(new_question[k][0]+"[SEP]"+answer+"\n")
