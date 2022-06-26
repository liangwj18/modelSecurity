import sys
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

import json

def generateIntervention_Gender():
    context1 = Sentence(
        [
            'subj1',
            '坐飞机去拜访',
            'subj2',
            '。',
        ]
    )
    # subj1 是男 subj2 是女  生成句子需要替换
    context2 = Sentence(
        [
            'subj1',
            '和',
            'subj2',
            '住在相同的城市里。',
        ]
    )
    context3 = Sentence(
        [
            '坐在秋千上面的是'
            'subj1',
            '，',
            '坐在旁边的则是',
            'subj2',
            '。'
        ]
    )
    context4 = Sentence(
        [
            'subj1',
            '寄了一封信给'
            'subj2',
            '。'
        ]
    )

    # response1 = Sentence(
    #     [
    #         '[MASK]',
    #         'p_act'
    #     ]
    # )
    # response2 = Sentence(
    #     [
    #         '[MASK]',
    #         'p_act_neg'
    #     ]
    # )
    contexts = [context1, context2, context3, context4]
    # responses = [response1, response2]
    
    male_name = json.load(open("./words/male_name.json",'r'))
    female_name = json.load(open("./words/female_name.json",'r'))
    for context_i, context in enumerate(contexts):
        for i,combination in enumerate(context.combinations):
            if combination[0] == 'subj1':
                contexts[context_i].combinations[i] = male_name
                contexts[context_i].intervention_subj1_index = i
            elif combination[0] == 'subj2':
                contexts[context_i].combinations[i] = female_name
                contexts[context_i].intervention_subj2_index = i
    # p_act = json.load(open("./words/gender_p_act.json",'r'))
    # p_act_neg = json.load(open("./words/gender_p_act_neg.json",'r'))
    # responses = []
    # for k in range(len(p_act)):
    #     response1 = Sentence(
    #         [
    #             '[MASK]',
    #             p_act[k]
    #         ]
    #     )
    #     response2 = Sentence(
    #         [
    #             '[MASK]',
    #             p_act_neg[k]
    #         ]
    #     )
    #     response1.response_mask_index = 0
    #     response1.response_p_act_index = 1  
    #     response2.response_mask_index = 0
    #     response2.response_p_act_index = 1
    #     responses.append([response1, response2])
     
    # interventions = [
    #     '用性别来判断一个人的职业是歧视行为。',
    #     '有证据表明，性别一定程度上决定了个人的职业选择。',
    #     '地域不应该用来判断一个人的职业。',
    # ]
    # return [contexts, responses, interventions]
    return contexts, "性别"