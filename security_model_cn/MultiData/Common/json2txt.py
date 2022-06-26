import json

js = json.load(open("prob_formats_5.json",'r'))

for a in range(100):
    print(js[a]['sentence'])