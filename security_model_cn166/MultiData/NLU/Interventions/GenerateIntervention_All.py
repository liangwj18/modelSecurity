import sys
import argparse
from GenerateIntervention_Education import generateIntervention_Education
from GenerateIntervention_Region import generateIntervention_Region
from GenerateIntervention_Gender import generateIntervention_Gender
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy
import json

def generateIntervention(contexts, questions, interventions, topics, output_file):

    res = []
    for context_sentence, subj_pair, topic in contexts:
        k = -1
        for kk in range(len(topics)):
            if topics[kk] != topic:
                k = kk
                break
        for question in questions:
            
            sentiment = 1
            if question['category'] == 'rudeness' or question['category'] == 'criminality':
                sentiment = -1
         
            y = [context_sentence + " " + question['question']]
            for i in range(3):
                intervention = interventions[question['category']][i].replace("[CMASK]",topic).replace("[NMASK]",topics[k])
                y.append(context_sentence + " " + intervention + question['question'])
            # print(y)
            x = {
                'ans':subj_pair,
                'sentiment':sentiment,
                'sentence':y,
                'topic':topic,
                'category':question['category']
            }
            res.append(x)
    json.dump(res,open(output_file,'w'))
            


if __name__ == "__main__":    


    # context change subj1 subj2 *2
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', default = './Interventions/interventions.json')
    args = parser.parse_args()
    
    contexts_all = []
    contexts_neg_all = []

    topics = []
    contexts_topic, topic = generateIntervention_Education()
    topics.append(topic)
    for context_sentence in contexts_topic:
        context_arr = context_sentence.generate_Intervention_format()
        for context_pair, subj_pair in context_arr:
            contexts_all.append([context_pair[0], subj_pair, topic])
            contexts_neg_all.append([context_pair[1], subj_pair, topic])

    contexts_topic, topic = generateIntervention_Education()
    for context_sentence in contexts_topic:
        context_arr = context_sentence.generate_Intervention_format()
        for context_pair, subj_pair in context_arr:
            contexts_all.append([context_pair[0], subj_pair, topic])
            contexts_neg_all.append([context_pair[1], subj_pair, topic])
    topics.append(topic)
    contexts_topic, topic = generateIntervention_Education()
    for context_sentence in contexts_topic:
        context_arr = context_sentence.generate_Intervention_format()
        for context_pair, subj_pair in context_arr:
            contexts_all.append([context_pair[0], subj_pair, topic])
            contexts_neg_all.append([context_pair[1], subj_pair, topic])
    topics.append(topic)
    questions_list = json.load(open("./words/questions.json", 'r'))
    interventions_list = json.load(open("./words/interventions.json",'r'))

    generateIntervention(contexts_all, questions_list, interventions_list, topics,args.output_file)