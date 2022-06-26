import json

f = json.load(open("corpus_input_new.json",'r'))


for i in range(10):
    print(f[i][0])
    # print("".join(f[i][0]))
    # print("".join(f[i][1]))
    # print("")
