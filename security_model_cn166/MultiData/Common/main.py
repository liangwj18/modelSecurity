import json

def todic(sentence, sentiment, topic, ans = []):
    return {
        'sentence':sentence,
        'sentiment':sentiment,
        'ans':ans,
        'topic':topic,
    }

def main():
    topics = ['Gender', 'Religion', 'Education', 'Profession']
    adj_with_subj = {}
    adj_with_subj['Gender_Profession'] = 1
    adj_with_subj['Religion_Profession'] = 1
    adj_with_subj['Education_Profession'] = 1

    subj2 = []
    # two subj
    for topic in topics:
        subj2.append([])
        # adj with subj
        for topic_2 in topics:
            if topic + "_" + topic_2 in adj_with_subj:
                topic_adj_subj = json.load(open("./Topic/{}/adj_like_subj.json".format(topic),'r'))
                topic_2_subj = json.load(open("./Topic/{}/subj.json".format(topic_2),'r'))
                for good_subj in topic_2_subj['group'][0]:
                    for pair in topic_adj_subj['pair']:
                        subj2[-1].append([pair[0]+good_subj, pair[1]+good_subj])
                    for good_adj in topic_adj_subj['group'][0]:
                        for bad_adj in topic_adj_subj['group'][1]:
                            subj2[-1].append([good_adj+good_subj, bad_adj+good_subj])
        topic_subj = json.load(open("./Topic/{}/subj.json".format(topic),'r'))
        for pair in topic_subj['pair']:
            subj2[-1].append(pair)
        for good_subj in topic_subj['group'][0]:
            for bad_subj in topic_subj['group'][1]:
                subj2[-1].append([good_subj, bad_subj])
    
    subj1 = []
    #one subj
    for topic in topics:
        subj1.append([[], []])
        # adj with subj
        for topic_2 in topics:
            if topic + "_" + topic_2 in adj_with_subj:
                topic_adj_subj = json.load(open("./Topic/{}/adj_like_subj.json".format(topic),'r'))
                topic_2_subj = json.load(open("./Topic/{}/subj.json".format(topic_2),'r'))
                for good_subj in topic_2_subj['group'][0]:
                    for pair in topic_adj_subj['pair']:
                        subj1[-1][0].append(pair[0]+good_subj)
                        subj1[-1][1].append(pair[1]+good_subj)
                    for good_adj in topic_adj_subj['group'][0]:
                        subj1[-1][0].append(good_adj+good_subj)
                    for bad_adj in topic_adj_subj['group'][1]:
                        subj1[-1][1].append(bad_adj+good_subj)    

        topic_subj = json.load(open("./Topic/{}/subj.json".format(topic),'r'))
        for pair in topic_subj['pair']:
            subj1[-1][0].append(pair[0])
            subj1[-1][1].append(pair[1])
        for good_subj in topic_subj['group'][0]:
            subj1[-1][0].append(good_subj)
        for bad_subj in topic_subj['group'][1]:
            subj1[-1][1].append(bad_subj)

    subj = [subj2, subj1]
    adjs = json.load(open("./adj.json",'r'))
    contexts = json.load(open("./context.json",'r'))

    # ??????????????????????????????????????????????????????????????????????????? ??????????????????????????????metric????????????????????????
    # ??????????????????
    # ???????????????????????? bias?????????  moral bias nlu   offsenive nlg
    # format1 format3 format4
    prob_formats = [[], [], [], [], []]
    for adj in adjs['group'][0]:
        for k in range(len(subj2)):
            for subj_pair in subj2[k]:
                #format1
                prob_formats[1-1].append(todic(sentence = subj_pair[0]+"???"+subj_pair[1]+"???"+adj+"??????", sentiment = 0, ans = ['??????','??????'], topic = topics[k]))
                for context in contexts['group'][0]:
                    context_rpr = context.replace('[subj1]', subj_pair[0]).replace('[subj2]', subj_pair[1])
                    # format3 format4
                    prob_formats[3-1].append(todic(sentence = context_rpr+";"+"????????????"+adj+"???", sentiment = 0, ans = subj_pair, topic = topics[k]))
                    prob_formats[4-1].append(todic(sentence = context_rpr+";"+"[MASK]?????????"+adj+".", sentiment = 0, ans = subj_pair, topic = topics[k]))

                    context_rpr = context.replace('[subj2]', subj_pair[0]).replace('[subj1]', subj_pair[1])
                    # format3 format4
                    prob_formats[3-1].append(todic(sentence = context_rpr+";"+"????????????"+adj+"???", sentiment = 0, ans = subj_pair, topic = topics[k]))
                    prob_formats[4-1].append(todic(sentence = context_rpr+";"+"[MASK]?????????"+adj+".", sentiment = 0, ans = subj_pair, topic = topics[k]))
    
    for adj in adjs['group'][1]:
        for k in range(4):
            for subj in subj1[k][1]:
                #format2
                prob_formats[2-1].append(todic(sentence = subj + "??????" + adj + "??????", sentiment = 0, ans = ['??????', '??????'], topic = topics[k]))
            for t in range(2):
                for subj in subj1[k][t]:
                    for good_adj in adjs['group'][0]:
                        #format5
                        prob_formats[5-1].append(todic(sentence = subj + "?????????[MASK]???", sentiment = t, ans = [good_adj, adj], topic = topics[k]))
                        # prob_formats[5-1].append(todic(sentence = subj_pair[1] + "?????????[MASK]???", sentiment = 1, ans = [good_adj, adj]))

    # generate_check format3
    # intervention format1 format2
    intervention = json.load(open("intervention.json",'r'))

    generate_formats = [[],[]]
    intervention_formats = [[],[]]
    verbs = json.load(open("verb.json",'r'))
    for verb in verbs['group'][1]:
        generate_formats[1].append(todic(sentence = "?????????????????????" + verb + "??????", sentiment = 1, topic = "Me"))
        for k in range(4):
            for subj in subj1[k][0] + subj1[k][1]:
                generate_formats[1].append(todic(sentence = "?????????"+subj+"?????????"+verb+"??????", sentiment = 1, topic = topics[k]))
            for subj_pair in subj2[k]:
                for context in contexts['group'][0]:
                    for x in range(2):

                        context_rpr = context.replace('[subj1]', subj_pair[x]).replace('[subj2]', subj_pair[x^1])
                        sentences_format1 = [context_rpr + "??????????????????????????????"]
                        sentences_format2 = [context_rpr + "???[MASK]????????????????????????"]
                        for i in range(3):
                            if i != 2:
                                inter = intervention[i].replace('[TOPIC]',topics[k])
                            else:
                                inter = intervention[i].replace('[TOPIC]',topics[(k+1)%4])
                            inter = inter.replace('[VERB]',verb)

                            sentences_format1.append(context_rpr + "???"+ inter+"??????????????????????????????")
                            sentences_format2.append(context_rpr + "???"+ inter+"???[MASK]????????????????????????")
                        intervention_formats[0].append(todic(sentence = sentences_format1, sentiment = 0, topic = topics[k], ans = subj_pair))
                        intervention_formats[1].append(todic(sentence = sentences_format2, sentiment = 0, topic = topics[k], ans = subj_pair))

    for adj in adjs['group'][1]:
        for k in range(4):
            for subj in subj1[k][0] + subj1[k][1]:
                generate_formats[0].append(todic(sentence = "????????????" + adj + "???" + subj+"??????????????????", sentiment = 1, topic = topics[k]))
    
    
    for i in range(len(prob_formats)):
        print("prob_formats len = {}".format(len(prob_formats[i])))
        json.dump(prob_formats[i], open("prob_formats_{}.json".format(i+1),'w'))
    for i in range(len(generate_formats)):
        print("generate_formats len = {}".format(len(generate_formats[i])))
        json.dump(generate_formats[i], open("generate_formats_{}.json".format(i+1),'w'))
    for i in range(len(intervention_formats)):
        print("intervention_formats len = {}".format(len(intervention_formats[i])))
        json.dump(intervention_formats[i], open("intervention_formats_{}.json".format(i+1),'w'))
if __name__ == "__main__":
    main()