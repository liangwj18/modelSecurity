from Models.CDial_GPT.DialModel_CDial_GPT import Args_CDial_GPT, DialModel_CDial_GPT
from Metric import NLG_Metric
# from utils.table import output_result_to_csv

import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--last_model_index', type = int, default = -1)
    parser.add_argument('--device', type = str, default = 'cuda:7')
    parser.add_argument('--SentenceSentiment_Analyzer_name', type = str, default = 'HuggingFace_techthiyanes')
    parser.add_argument('--SentenceAttack_Analyzer_name', type = str, default = 'Translate_bert_en')    
    parser.add_argument('--Career_and_Family_word_dir', type = str, default = "./Analyzer/CareerAndFamily")
    parser.add_argument('--reGenerateResponse', action = "store_true")
    parser.add_argument('--debug', action = "store_true")
    args = parser.parse_args()

    models = [
        [DialModel_CDial_GPT(Args_CDial_GPT(device = args.device)), 'CDial-GPT'],
    ]
    
    Topics = [
        # 'Gender',
        'SunHao'
    ]

    NLG_Metrics = [
        'OpposeSentiment',
        'Politeness',
        'Diversity',
        'Career_and_Family'
    ]

    # Data_path = "../../MultiData/NLG"
    Data_path = "../../MultiData/NLU/SunHao/"
    result_dir = "./metric_record"
    model_names = []

    for model_index, model in enumerate(models):
        model, model_name = model
        model_names.append(model_name)
        if model_index <= args.last_model_index:
            continue
        for topic in Topics:
            # topic_data_path = os.path.join(Data_path, topic, "corpus_input.json")
            topic_data_path = os.path.join(Data_path, topic, "all_nlg.json")
            assert os.path.exists(topic_data_path) == True
            metric = NLG_Metric(
                nlg_model= model,
                nlg_model_name = model_name,
                dataset_path = topic_data_path,
                dataset_name = topic,
                output_dir = result_dir,
                device = args.device,
            )
            print("Generating Response...", topic_data_path)
            metric.Generate_Response(args)

            for Metric_name in NLG_Metrics:
                ret = metric.score(Metric_name, args)
                assert ret != -1

    output_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.csv"))

if __name__ == "__main__":
    main()
    