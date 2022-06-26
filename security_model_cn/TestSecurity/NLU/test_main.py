from Metric import Metric
from utils.table import output_result_to_csv, output_intervention_result_to_csv

import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--last_model_index', type = int, default = -1)
    parser.add_argument('--device', type = str, default = "cuda")
    args = parser.parse_args()

    models = [
        ['bert-base-chinese', 'bert-base', True],
        ['hfl/chinese-bert-wwm-ext', 'bert-hfl-wwm-base', True],
        ['hfl/chinese-roberta-wwm-ext', 'roberta-hfl-wwm-base', True],
        ['hfl/chinese-roberta-wwm-ext-large', 'roberta-hfl-wwm-large', True],
        ['nghuyong/ernie-1.0', 'ernie-baidu-base', True],
        # ['tsinghua/ernie','ernie-tsinghua-base',False], 英文
        # ['hfl/chinese-xlnet-base', 'xlnet-hfl-base', True],
        ['ckiplab/albert-tiny-chinese','albert-ckiplab-tiny', True],
    ]

    Topics = [
        # 'Education',
        # 'Profession',
        # 'Gender',
        # 'Region',
        'Interventions'
    ]

    Metrics = [
        # 'SS',
        # 'CrowsP'
        'interventions'
    ]

    Data_path = "../../MultiData/NLU"
    result_dir = "./metric_record"
    model_names = []

    for model_index, model in enumerate(models):
        model_transformers_name, model_name, in_transformer = model
 
        model_names.append(model_name)
        # if model_index > 0:
        # continue
        if model_index <= args.last_model_index:
            continue
        for topic in Topics:
            for Metric_name in Metrics:
                topic_data_path = os.path.join(Data_path, topic, topic, Metric_name+".json")
                print(topic_data_path)
                assert os.path.exists(topic_data_path) == True
                metric = Metric(
                    model_transformers_name = model_transformers_name,
                    model_name = model_name,
                    data_path = topic_data_path,
                    dataset_name = topic,
                    output_dir = result_dir,
                    device = args.device,
                )
                ret = metric.score(Metric_name)
                assert ret != -1

    # output_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.csv"))
    output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, os.path.join(result_dir, "table.md"))

if __name__ == "__main__":
    main()
    