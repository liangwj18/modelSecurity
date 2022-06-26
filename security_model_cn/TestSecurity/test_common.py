# from NLU.Metric import Metric
from NLU.utils.table import output_result_to_csv, output_intervention_result_to_csv,output_Prob_format_to_csv
# from NLU.utils.table import output_Generate_format_0_to_csv

import os
import argparse

import torch
import torch.nn.functional as F
def test_nlu(args):
    
    models = [
        ['bert-base-chinese', 'bert-base', True],
        ['hfl/chinese-bert-wwm-ext', 'bert-hfl-wwm-base', True],
        ['hfl/chinese-roberta-wwm-ext', 'roberta-hfl-wwm-base', True],
        ['hfl/chinese-roberta-wwm-ext-large', 'roberta-hfl-wwm-large', True],
        ['nghuyong/ernie-1.0', 'ernie-baidu-base', True],
        # ['tsinghua/ernie','ernie-tsinghua-base',False]
        # ['hfl/chinese-xlnet-base', 'xlnet-hfl-base', True],
        ['ckiplab/albert-tiny-chinese','albert-ckiplab-tiny', True],
    ]
    
    DataSets = [
        'prob_formats_4.json',
        'prob_formats_5.json',
        # 'intervention_formats_2.json'
    ]
 
    
    model_names = []

    for model_index, model in enumerate(models):
        model_transformers_name, model_name, in_transformer = model 
        model_names.append(model_name)
        continue
        for dataset_file in DataSets:
            
            dataset_path = os.path.join(args.datasets_dir, dataset_file)    
            assert os.path.exists(dataset_path) == True
            metric = Metric(
                model_transformers_name = model_transformers_name,
                model_name = model_name,
                data_path = dataset_path,
                dataset_name = dataset_file.split(".json")[0],
                output_dir = args.output_dir,
                device = args.device,
            )
            ret = metric.score(dataset_file.split(".json")[0])
            assert ret != -1
    # output_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.csv"))
    # output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.md"))
    output_Prob_format_to_csv(model_names, DataSets, args.output_dir, os.path.join(args.output_dir, "nlu_table.md"))

# from NLG.Models.CDial_GPT.DialModel_CDial_GPT import Args_CDial_GPT, DialModel_CDial_GPT
# from NLG.Models.EVA.DialModel_Eva import DialModel_Eva, DialModel_Eva2
# from NLG.Models.CPM.DialModel_CPM import DialModel_CPM
from NLG.Models.Zhouwenwang.DialModel_Zhouwenwang import DialModel_Zhouwenwang
from NLG.Metric import NLG_Metric
def test_nlg(args):
    models = [
        # [DialModel_CDial_GPT(Args_CDial_GPT(device = args.device)), 'CDial-GPT'],
        # [DialModel_Eva(args.device),'Eva'],
        # [DialModel_Eva2(args.device),'Eva2']
        # [DialModel_CPM(args.device), 'CPM'],
        [DialModel_Zhouwenwang(args.device),'Zhouwenwang']
    ]
    # models[0][0].test()
    # assert 1==2
    DataSets = [
        # 'prob_formats_1.json',
        # 'prob_formats_2.json',
        # 'prob_formats_3.json',
        # 'generate_formats_0.json',
        # 'generate_formats_1.json',
        # 'generate_formats_2.json',
        # 'intervention_formats_1.json',
        # 'question_not_in_round2_and_3.json',
        'question.json'
        # 'prob_formats_1_gen.json'
        # 'prob_formats_1_new.json'
        # 'prob_formats_2_new.json'
        # 'prob_formats_1_vec.json'
        # 'prob_formats_3_new.json'
        # 'prob_formats_6_new.json'
    ]

    # None_prompt = ['[MASK]','N/A', " "]
    None_prompt = []

    model_names = []

    for model_index, model in enumerate(models):
        model, model_name = model
        # model = model.to(args.device)
        model_names.append(model_name)
        # continue
        for dataset_file in DataSets:
            
            dataset_path = os.path.join(args.datasets_dir, dataset_file)    
            assert os.path.exists(dataset_path) == True
            metric = NLG_Metric(
                nlg_model= model,
                nlg_model_name = model_name,
                dataset_path = dataset_path,
                dataset_name = dataset_file.split(".json")[0],
                output_dir = args.output_dir,
                device = args.device,
                None_prompt = None_prompt,
            )
            # return 
            ret = metric.score(dataset_file.split(".json")[0], args)
            assert ret != -1
            # topic_data_path = os.path.join(Data_path, topic, "corpus_input.json")
            
            # print("Generating Response...", topic_data_path)
            # metric.Generate_Response(args)

            # for Metric_name in NLG_Metrics:
            #     ret = metric.score(Metric_name, args)
            #     assert ret != -1
    # prob_datasets = [
    #     'prob_formats_1.json',
    #     'prob_formats_2.json',
    #     'prob_formats_3.json',
    # ]
    # for dataset in prob_datasets:
    #     x = dataset.split(".json")[0]
    #     pa = os.path.join(args.output_dir,"Eva-{}".format(x),x+".txt")
    #     pa1 = os.path.join(args.output_dir,"Eva-{}".format(x),x+"1.txt")
    #     # with open(pa1,'w') as ff:
    #     with open(pa,'r') as f:
    #         lines = f.readlines()
    #         sum_a = 0
    #         sum_t = 0

    #         for i in range(0, len(lines)-1):
    #             if i%3 == 1:
    #                 line = lines[i].split("\n")[0]
    #                 a = float(line.split(' ')[-2])
    #                 b = float(line.split(' ')[-1])
    #                 # a = F.softmax(torch.tensor([a,b]),dim = 0).detach().numpy()
    #                 sum_a += a - b
    #                 sum_t += 1
    #                 # ff.write(" ".join(line.split(' ')[:-2])+" "+ str(a[0])+ " "+str(a[1])+"\n")
    #             # else:
    #                 # ff.write(lines[i]+"\n")
    #         print(pa, sum_a/sum_t)
    # output_Prob_format_to_csv(model_names,prob_datasets, result_dir, os.path.join(result_dir, "nlg_prob.md"))
    # output_Generate_format_0_to_csv(model_names,['generate_formats_0.json'],result_dir,os.path.join(result_dir, "nlg_generate_0.md"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type = str, default = "cuda:0")
    #用export
    parser.add_argument('--output_dir', type = str, default = "./common_result")
    parser.add_argument('--datasets_dir', type = str, default = "../MultiData/Common")
    parser.add_argument('--SentenceSentiment_Analyzer_name', type = str, default = 'HuggingFace_techthiyanes')
    parser.add_argument('--SentenceAttack_Analyzer_name', type = str, default = 'COLD_Detector')    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # test_nlu(args)
    test_nlg(args)
    # output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.md"))

if __name__ == "__main__":
    main()
# from NLU.Metric import Metric
# from NLU.utils.table import output_result_to_csv, output_intervention_result_to_csv,output_Prob_format_to_csv
# # from NLU.utils.table import output_Generate_format_0_to_csv

# import os
# import argparse

# import torch
# import torch.nn.functional as F
# def test_nlu(args):
    
#     models = [
#         ['bert-base-chinese', 'bert-base', True],
#         ['hfl/chinese-bert-wwm-ext', 'bert-hfl-wwm-base', True],
#         ['hfl/chinese-roberta-wwm-ext', 'roberta-hfl-wwm-base', True],
#         ['hfl/chinese-roberta-wwm-ext-large', 'roberta-hfl-wwm-large', True],
#         ['nghuyong/ernie-1.0', 'ernie-baidu-base', True],
#         # ['tsinghua/ernie','ernie-tsinghua-base',False]
#         # ['hfl/chinese-xlnet-base', 'xlnet-hfl-base', True],
#         ['ckiplab/albert-tiny-chinese','albert-ckiplab-tiny', True],
#     ]
    
#     DataSets = [
#         'prob_formats_4.json',
#         'prob_formats_5.json',
#         # 'intervention_formats_2.json'
#     ]
 
    
#     model_names = []

#     for model_index, model in enumerate(models):
#         model_transformers_name, model_name, in_transformer = model 
#         model_names.append(model_name)
#         continue
#         for dataset_file in DataSets:
            
#             dataset_path = os.path.join(args.datasets_dir, dataset_file)    
#             assert os.path.exists(dataset_path) == True
#             metric = Metric(
#                 model_transformers_name = model_transformers_name,
#                 model_name = model_name,
#                 data_path = dataset_path,
#                 dataset_name = dataset_file.split(".json")[0],
#                 output_dir = args.output_dir,
#                 device = args.device,
#             )
#             ret = metric.score(dataset_file.split(".json")[0])
#             assert ret != -1
#     # output_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.csv"))
#     # output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.md"))
#     output_Prob_format_to_csv(model_names, DataSets, args.output_dir, os.path.join(args.output_dir, "nlu_table.md"))

# from NLG.Models.CDial_GPT.DialModel_CDial_GPT import Args_CDial_GPT, DialModel_CDial_GPT
# from NLG.Models.EVA.DialModel_Eva import DialModel_Eva, DialModel_Eva2
# from NLG.Models.CPM.DialModel_CPM import DialModel_CPM
# from NLG.Metric import NLG_Metric
# def test_nlg(args):
#     models = [
#         # [DialModel_CDial_GPT(Args_CDial_GPT(device = args.device)), 'CDial-GPT'],
#         # [DialModel_Eva(args.device),'Eva'],
#         # [DialModel_Eva2(args.device),'Eva2'],
#         # [DialModel_CPM(args.device), 'CPM'],
#     ]
#     DataSets = [
#         # 'prob_formats_1.json',
#         # 'prob_formats_2.json',
#         # 'prob_formats_3.json',
#         # 'generate_formats_0.json',
#         # 'generate_formats_1.json',
#         # 'generate_formats_2.json',
#         # 'intervention_formats_1.json',
#         # 'question.json',
#     ]

#     # None_prompt = ['[MASK]','N/A', " "]

#     model_names = []
#     print("FUCK")

#     for model_index, model in enumerate(models):
#         model, model_name = model
#         model_names.append(model_name)
#         # continue
#         for dataset_file in DataSets:
            
#             dataset_path = os.path.join(args.datasets_dir, dataset_file)    
#             assert os.path.exists(dataset_path) == True
#             metric = NLG_Metric(
#                 nlg_model= model,
#                 nlg_model_name = model_name,
#                 dataset_path = dataset_path,
#                 dataset_name = dataset_file.split(".json")[0],
#                 output_dir = args.output_dir,
#                 device = args.device,
#                 None_prompt = None_prompt,
#             )
#             # return 
#             ret = metric.score(dataset_file.split(".json")[0], args)
#             assert ret != -1
#             # topic_data_path = os.path.join(Data_path, topic, "corpus_input.json")
            
#             # print("Generating Response...", topic_data_path)
#             # metric.Generate_Response(args)

#             # for Metric_name in NLG_Metrics:
#             #     ret = metric.score(Metric_name, args)
#             #     assert ret != -1
#     # prob_datasets = [
#     #     'prob_formats_1.json',
#     #     'prob_formats_2.json',
#     #     'prob_formats_3.json',
#     # ]
#     # for dataset in prob_datasets:
#     #     x = dataset.split(".json")[0]
#     #     pa = os.path.join(args.output_dir,"Eva-{}".format(x),x+".txt")
#     #     pa1 = os.path.join(args.output_dir,"Eva-{}".format(x),x+"1.txt")
#     #     # with open(pa1,'w') as ff:
#     #     with open(pa,'r') as f:
#     #         lines = f.readlines()
#     #         sum_a = 0
#     #         sum_t = 0

#     #         for i in range(0, len(lines)-1):
#     #             if i%3 == 1:
#     #                 line = lines[i].split("\n")[0]
#     #                 a = float(line.split(' ')[-2])
#     #                 b = float(line.split(' ')[-1])
#     #                 # a = F.softmax(torch.tensor([a,b]),dim = 0).detach().numpy()
#     #                 sum_a += a - b
#     #                 sum_t += 1
#     #                 # ff.write(" ".join(line.split(' ')[:-2])+" "+ str(a[0])+ " "+str(a[1])+"\n")
#     #             # else:
#     #                 # ff.write(lines[i]+"\n")
#     #         print(pa, sum_a/sum_t)
#     # output_Prob_format_to_csv(model_names,prob_datasets, result_dir, os.path.join(result_dir, "nlg_prob.md"))
#     # output_Generate_format_0_to_csv(model_names,['generate_formats_0.json'],result_dir,os.path.join(result_dir, "nlg_generate_0.md"))

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type = str, default = "cuda:0")
#     #用export
#     parser.add_argument('--output_dir', type = str, default = "./common_result")
#     parser.add_argument('--datasets_dir', type = str, default = "../MultiData/Common")
#     parser.add_argument('--SentenceSentiment_Analyzer_name', type = str, default = 'HuggingFace_techthiyanes')
#     parser.add_argument('--SentenceAttack_Analyzer_name', type = str, default = 'COLD_Detector')    
#     args = parser.parse_args()

#     if not os.path.exists(args.output_dir):
#         os.mkdir(args.output_dir)
#     # test_nlu(args)
#     test_nlg(args)
#     # output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.md"))

# if __name__ == "__main__":
#     main()
    