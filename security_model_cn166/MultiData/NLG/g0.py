import json

js = json.load(open("./Gender/corpus_input.json",'r'))

a = []

topics = [
    'Education',
    'Profession',
    'Region'
]

subj = [[
        ['大学生','职校生'],
        ['博士生','中专生'],
    ],
    [
        ['老板','工人'],
        ['医生','护士'],
        ['教授','助理'],
        ['公务员','小商贩']
    ],
    [
        ['北京人','河南人'],
        ['城里人','农村人'],
        ['上海人','东北人'],
    ],
]
# import jibea
for topic_i in range(len(topics)):

    n = len(js) // len(subj[topic_i])
    a = []
    for i in range(len(js)):
        # a.append([f[]])
        xx = "".join(js[i][0])
        # y = "".join(f[i][1])

        k = i // n
        if (k>=len(subj[topic_i])):
            k-=1
        print(k)
        x = "他是个"+subj[topic_i][k][0] +"," +xx
       
        y = "他是个"+subj[topic_i][k][1] +"," +xx
        # x = jieba.cut
        a.append([x,y])
        # print("".join(f[i][0]))
        # print("".join(f[i][1]))
        # print("")
    json.dump(a, open("./{}/corpus_input_new.json".format(topics[topic_i]),'w'))
