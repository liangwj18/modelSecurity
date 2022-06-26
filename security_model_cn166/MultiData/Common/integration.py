import json

topics = ['Gender','Region','Education','Profession']

integration = {'topics':topics, 'data':[],'specific_word':[]}
for topic in topics:
    integration['data'].append(json.load(open('../NLG/{}/corpus_input_new.json'.format(topic),'r'))[:12000])
    print(len(integration['data'][-1]))
    integration['specific_word'].append([
        json.load(open("../NLG/{}/specific_word/good.json".format(topic),'r')),
        json.load(open("../NLG/{}/specific_word/bad.json".format(topic),'r')),
    ])


json.dump(integration, open("generate_formats_0_new.json",'w'))