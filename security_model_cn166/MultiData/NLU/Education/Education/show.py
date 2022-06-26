import json

f = json.load(open("all.json",'r'))


for i in range(10):
    print(f[i])
