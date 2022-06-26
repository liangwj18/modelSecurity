import os

def output_result_to_csv(model_names, Topics, Metrics, result_dir, output_path):

    print(output_path)
    f = open(output_path, 'w', encoding = 'gbk')
    f.write("分数 = 反刻板印象的比例 50为best,")
    for topic_index, topic in enumerate(Topics):
        f.write(topic)
        if topic_index + 1 < len(Topics):
            f.write(','*(len(Metrics)-1))
        f.write(',')
    f.write('\n')
    f.write('Model_Name')
    for topic_index in range(len(Topics)):
        for metric in Metrics:
            f.write(",")
            f.write(metric)
    f.write("\n")

    for model_name in model_names:
        f.write(model_name)
        for topic in Topics:
            for metric in Metrics:
                file_path = os.path.join(result_dir, model_name+"-"+topic, metric+".txt")
                with open(file_path, 'r') as file_f:
                    value = round(float(file_f.readlines()[-1]), 2)
                    f.write(",")
                    f.write(str(value))
        f.write("\n")



def output_intervention_result_to_csv(model_names, Topics, Metrics, result_dir, output_path):

    print(output_path)
    f = open(output_path, 'w', encoding = 'gbk')
    f.write("# Intervention Result\n")
    f.write("|Model Name|Model Base Bias|Lower Bias|Upper Bias|Unrelated Bias|\n")
    f.write("|----|----|----|----|----|\n")

    for model_name in model_names:
        f.write("|{}|".format(model_name))
        for topic in Topics:
            for metric in Metrics:
                file_path = os.path.join(result_dir, model_name+"-"+topic, "intervention"+".txt")
                with open(file_path, 'r') as file_f:
                    line = file_f.readlines()[-1]
                    print(line)
                    p = 0
                    for i in range(4):
                        t = line.find('.', p)
                        st = line[t-1:t+5]
                        st = st.split(" ")[0]
                        st = st.split(":")[0]
                        value = round(float(st),4)
                        p = t + 2
                        f.write("{}|".format(str(value)))
        f.write("\n")


def output_Prob_format_to_csv(model_names, DataSets, result_dir, output_path):

    print(output_path)
    f = open(output_path, 'w', encoding = 'gbk')
    f.write("# NLU Result\n")
    f.write("|Metric_name|")
    for model_name in model_names:
        f.write("{}|".format(model_name))
    f.write("\n")
    f.write("|"+"----|"*(len(model_names)+1)+"\n")

    DataSets = DataSets[:2]
    for dataset in DataSets:
        metric = dataset.split(".json")[0]

        for i in range(2):
            topic = "avg_bias_prob" if i == 0 else "avg_bigger_prob_rate"
            f.write("|{}|".format(metric+"_"+topic))
            for model_name in model_names:
                file_path = os.path.join(result_dir, model_name+"-"+metric, metric+".txt")
                print(file_path)
                with open(file_path, 'r') as file_f:
                    # line = file_f.readlines()[-1].split("\n")[0]
                    lines = file_f.readlines()
                    sum_a = 0
                    sum_t = 0
                    print(len(lines))
                    for k in range(0, len(lines)-1, 4):
                        line = lines[k+1].split("\n")[0]
                        a = float(line.split(' ')[-1])
                        line = lines[k+2].split("\n")[0]
                        b = float(line.split(' ')[-1])
                        sum_a += a-b
                        sum_t += 1
                    # print(pa, sum_a/sum_t)
                    # avg_prob = round(float(line.split('avg_prob = ')[1].split(',')[0]),3)
                    avg_prob = sum_a/sum_t
                    avg_bigger_count = round(float(lines[-1].split('avg_bigger_count = ')[1].split('%')[0])/100,3)
                    if i == 0:
                        f.write("{}|".format(avg_prob))
                    else:
                        f.write("{}|".format(avg_bigger_count))
            f.write("\n")
    

# def output_Generate_format_0_to_csv(model_names, DataSets, result_dir, output_path):

#     print(output_path)
#     f = open(output_path, 'w', encoding = 'gbk')
#     f.write("# NLU Result\n")
#     f.write("|Metric_name|")
#     for model_name in model_names:
#         f.write("{}|".format(model_name))
#     f.write("\n")
#     f.write("|"+"----|"*(len(model_names)+1)+"\n")

#     DataSets = DataSets[:2]
#     for dataset in DataSets:
#         metric = dataset.split(".json")[0]

#         for i in range(2):
#             topic = "avg_bias_prob" if i == 0 else "avg_bigger_prob_rate"
#             f.write("|{}|".format(metric+"_"+topic))
#             for model_name in model_names:
#                 file_path = os.path.join(result_dir, model_name+"-"+metric, metric+".txt")
#                 with open(file_path, 'r') as file_f:
#                     line = file_f.readlines()[-1].split("\n")[0]
#                     avg_prob = float(line.split('avg_prob = ')[1].split(',')[0])
#                     avg_bigger_count = float(line.split('avg_bigger_count = ')[1].split('%')[0])/100
#                     if i == 0:
#                         f.write("{}|".format(avg_prob))
#                     else:
#                         f.write("{}|".format(avg_bigger_count))
#             f.write("\n")
    