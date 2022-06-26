import sys
import argparse
from GenerateGender_BetterWord import generateGender_BetterWord
from GenerateGender_NormalWord import generateGender_NormalWord
from GenerateGender_WorseWord import generateGender_WorseWord
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("output_type", choices = ['SS','CrowsP','all'])
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    genDataGender = GenerateDataBase("Gender", args.output_type, args.debug)
    
    sentences = generateGender_BetterWord()
    for sentence in sentences:
        genDataGender.add(sentence)

    if args.output_type != 'SS':
        sentences = generateGender_NormalWord()
        for sentence in sentences:
            genDataGender.add(sentence)

    sentences = generateGender_WorseWord()
    for sentence in sentences:
        genDataGender.add(sentence)

    if args.output_type == "CrowsP":
        genDataGender.generate_CrowsP()
    if args.output_type == "SS":
        genDataGender.generate_SS()
    if args.output_type == "all":
        genDataGender.generate()