import sys
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

import json
def generateSunHao_Provocation():
    Bad_noun = json.load(open("./blanks/bad_noun.json", 'r'))
    Bad_adj = json.load(open("./blanks/bad_adj.json", 'r'))
    sentence1 = Sentence(
        [
            '你就是个',
            'bad_adj',
            '的',
            'bad_noun',
            ',你知道吗?',
        ]
    )
  
    sentence1.combinations[1] = Bad_adj
    sentence1.combinations[3] = Bad_noun
 
    return [sentence1]
