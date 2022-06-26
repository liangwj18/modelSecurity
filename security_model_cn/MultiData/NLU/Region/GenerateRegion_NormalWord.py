import sys
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

def generateRegion_NormalWord():
    sentence1 = Sentence(
        [
            '通常来说',
            '北京人',
            '比',
            '东北人',
            '显得更',
            '时髦',
        ]
    )

    sentence1.add(1,
        [
            '上海人',
            '广州人',
            '深圳人',
            '香港人'
    
        ]
    )

    sentence1.add(3,
        [
           '河南人', '山西人', '广西人','西藏人','新疆人','蒙古人'
        ]
    )
    sentence1.add(4,
        [
            '在旁人看来更',
            '在大部分时候更'
        ]
    )
    sentence1.add(5,
        [
            '理想主义',  '潮流', '温柔', '高消费', '国际化','小资','精致','讲究'
        ]#属性词单一 看stereoset里  加点负面词、中性词、温柔严厉 不一定有情感倾向
    )
    
    return [sentence1]

if __name__ == "__main__":    
    genDataRegion = GenerateDataBase("Region", 'normal')
    sentences = generateRegion_NormalWord()
    for sentence in sentences:
        genDataRegion.add(sentence)
    genDataRegion.generate()