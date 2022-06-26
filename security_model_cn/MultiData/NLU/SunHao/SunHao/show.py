import json

a = json.load(open("./all.json",'r'))
b = []
for x in a[0]:
    y = [x, x]
    b.append(y)
json.dump(b, open("all_nlg.json",'w'))