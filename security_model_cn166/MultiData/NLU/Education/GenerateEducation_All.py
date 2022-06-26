import sys
import argparse
from GenerateEducation_BetterWord import generateEducation_BetterWord
from GenerateEducation_NormalWord import generateEducation_NormalWord
from GenerateEducation_WorseWord import generateEducation_WorseWord
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("output_type", choices = ['SS','CrowsP','all'])
    parser.add_argument('--device', type = str, default = 'cuda:7')
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    genDataEducation = GenerateDataBase(args, "Education", args.output_type, args.debug)
    
    sentences = generateEducation_BetterWord()
    for sentence in sentences:
        genDataEducation.add(sentence)

    if args.output_type != 'SS':
        sentences = generateEducation_NormalWord()
        for sentence in sentences:
            genDataEducation.add(sentence)

    sentences = generateEducation_WorseWord()
    for sentence in sentences:
        genDataEducation.add(sentence)

    if args.output_type == "CrowsP":
        genDataEducation.generate_CrowsP()
    if args.output_type == "SS":
        genDataEducation.generate_SS()
    if args.output_type == "all":
        genDataEducation.generate()