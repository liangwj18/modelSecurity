import json

f = json.load(open("corpus_input.json",'r'))

x = []
for i in range(len(f)):
    x.append(["".join(f[i][0]),"".join(f[i][1])])
    # print("".join(f[i][0]))
    # print("".join(f[i][1]))
    # print("")
json.dump(x,open("corpus_input_new.json",'w'))
