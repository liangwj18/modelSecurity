import sys
import argparse
from GenerateProfession_BetterWord import generateProfession_BetterWord
from GenerateProfession_NormalWord import generateProfession_NormalWord
from GenerateProfession_WorseWord import generateProfession_WorseWord
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("output_type", choices = ['SS','CrowsP','all'])
    parser.add_argument('--debug', action = 'store_true')
    args = parser.parse_args()

    genDataProfession = GenerateDataBase("Profession", args.output_type, args.debug)
    
    sentences = generateProfession_BetterWord()
    for sentence in sentences:
        genDataProfession.add(sentence)

    if args.output_type != 'SS':
        sentences = generateProfession_NormalWord()
        for sentence in sentences:
            genDataProfession.add(sentence)

    sentences = generateProfession_WorseWord()
    for sentence in sentences:
        genDataProfession.add(sentence)

    if args.output_type == "CrowsP":
        genDataProfession.generate_CrowsP()
    if args.output_type == "SS":
        genDataProfession.generate_SS()
    if args.output_type == "all":
        genDataProfession.generate()