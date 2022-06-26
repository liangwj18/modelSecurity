import sys
import argparse
from GenerateSunHao_UnfairGeneralization import generateSunHao_UnfairGeneralization
from GenerateSunHao_Comparison import generateSunHao_Comparison
from GenerateSunHao_Moral import generateSunHao_Moral
from GenerateSunHao_SelfExposure import generateSunHao_SelfExposure
from GenerateSunHao_Provocation import generateSunHao_Provocation
sys.path.append("../")
from GenerateDataBase import GenerateDataBase
from Sentence import Sentence
import copy

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('output_type', choices = ['all', 'comparison', 'moral', 'provocation', 'selfexposure', 'unfairgeneralization'])
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--device', default = 'cuda:7')
    args = parser.parse_args()

    genDataSunHao = GenerateDataBase(args, "SunHao", args.output_type, args.debug)
    
    if args.output_type == 'all' or args.output_type == 'comparison':
        sentences = generateSunHao_Comparison()
        for sentence in sentences:
            genDataSunHao.add(sentence)

    if args.output_type == 'all' or args.output_type == 'moral':
        sentences = generateSunHao_Moral()
        for sentence in sentences:
            genDataSunHao.add(sentence)
    if args.output_type == 'all' or args.output_type == 'provocation':
        sentences = generateSunHao_Provocation()
        for sentence in sentences:
            genDataSunHao.add(sentence)
    
    if args.output_type == 'all' or args.output_type == 'selfexposure':
        sentences = generateSunHao_SelfExposure()
        for sentence in sentences:
            genDataSunHao.add(sentence)
    
    if args.output_type == 'all' or args.output_type == 'unfairgeneralization':
        sentences = generateSunHao_UnfairGeneralization()
        for sentence in sentences:
            genDataSunHao.add(sentence)
    genDataSunHao.generate()