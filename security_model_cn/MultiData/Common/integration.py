import json

topics = ['Gender']

integration = {'topics':topics, 'data':[],'specific_word':[]}
for topic in topics:
    integration['data'].append(json.load(open('../NLG/{}/corpus_input.json'.format(topic),'r')))
    print(len(integration['data'][0]))
    integration['specific_word'].append([
        json.load(open("../NLG/{}/specific_word/good.json".format(topic),'r')),
        json.load(open("../NLG/{}/specific_word/bad.json".format(topic),'r')),
    ])


# json.dump(integration, open("generate_formats_0.json",'w'))