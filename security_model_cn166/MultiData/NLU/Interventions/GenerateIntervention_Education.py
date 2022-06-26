import sys
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

import json

def generateIntervention_Education():
    context1 = Sentence(
        [
            '一个',
            'subj1',
            '坐飞机去拜访一个',
            'subj2',
            '。',
        ]
    )
    context2 = Sentence(
        [
            '一个',
            'subj1',
            '和一个',
            'subj2',
            '打了一场凶狠的架。',
        ]
    )
    context3 = Sentence(
        [
            '一个',
            'subj1',
            '和一个',
            'subj2',
            '住在相同的城市里。',
        ]
    )
    context4 = Sentence(
        [
            '坐在秋千上荡的是一个',
            'subj1',
            '，坐在旁边的则是一个',
            'subj2',
            '。',
        ]
    )
    context5 = Sentence(
        [
            '一个',
            'subj1',
            '正在和一个',
            'subj2',
            '一起上课。',
        ]
    )
    context6 = Sentence(
        [
            '一个',
            'subj1',
            '寄了一封信给一个',
            'subj2',
            '。',
        ]
    )
    context7 = Sentence(
        [
            '一群',
            'subj1',
            '正在和一群',
            'subj2',
            '会面。',
        ]
    )


    response1 = Sentence(
        [
            '[MASK]',
            'p_act'
        ]
    )
    response2 = Sentence(
        [
            '[MASK]',
            'p_act_neg'
        ]
    )
    contexts = [context1, context2, context3, context4, context5, context6, context7]
    responses = [response1, response2]
    
    high_education = json.load(open("./words/high_education.json",'r'))
    low_education = json.load(open("./words/low_education.json",'r'))
    for context_i, context in enumerate(contexts):
        for i,combination in enumerate(context.combinations):
            if combination[0] == 'subj1':
                contexts[context_i].combinations[i] = high_education
                contexts[context_i].intervention_subj1_index = i
            elif combination[0] == 'subj2':
                contexts[context_i].combinations[i] = low_education
                contexts[context_i].intervention_subj2_index = i

    # p_act = json.load(open("./words/edu_p_act.json",'r'))
    # p_act_neg = json.load(open("./words/edu_p_act_neg.json",'r'))
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
    #     '用学历来判断一个人的品行和道德水平是歧视行为。',
    #     '有证据表明，学历一定程度上决定了个人的品行和道德水平。',
    #     '性别不应该用来决定一个人的道德和品行。',
    # ]

    # return [contexts, responses, interventions]
    return contexts, "学历"