import sys
import argparse
from GenerateRegion_BetterWord import generateRegion_BetterWord
from GenerateRegion_NormalWord import generateRegion_NormalWord
from GenerateRegion_WorseWord import generateRegion_WorseWord
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("output_type", choices = ['SS','CrowsP','all'])
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    genDataRegion = GenerateDataBase("Region", args.output_type, args.debug)
    
    sentences = generateRegion_BetterWord()
    for sentence in sentences:
        genDataRegion.add(sentence)

    if args.output_type != 'SS':
        sentences = generateRegion_NormalWord()
        for sentence in sentences:
            genDataRegion.add(sentence)

    sentences = generateRegion_WorseWord()
    for sentence in sentences:
        genDataRegion.add(sentence)

    if args.output_type == "CrowsP":
        genDataRegion.generate_CrowsP()
    if args.output_type == "SS":
        genDataRegion.generate_SS()
    if args.output_type == "all":
        genDataRegion.generate()