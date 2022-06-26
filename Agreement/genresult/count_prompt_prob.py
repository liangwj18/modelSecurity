import json
import numpy as np
import os

def divsum(group):
    x = np.expand_dims(np.sum(np.array(group),axis = 1), axis = 1)
    return (np.array(group)/np.pad(x,((0,0),(0,1)),'edge')).tolist()

def divsum_all(group):
    x = np.array(group).reshape(-1,1).squeeze(1)
    return np.expand_dims(x / np.sum(x,axis = 0), axis = 1).reshape(-1, 2).tolist()

def output(array, name):
    # print("|DataName|0bias|1bias|tbias|0count|1count|tcount|")
    # print("|----|----|----|----|----|----|----|")
    # for k in range(len(array[0])):

    #     bias = [0, 0]
    #     count = [0, 0]
    #     for i in range(len(array)):
    #         bias[i%2] += array[i][k][0]

    #     print("|{}|{}|{}|{}|{}|{}|".format())
    data = [[], []]
    for i in range(len(array)):
        data[i%2].append(array[i])
    
    print("---{}---".format(name))
    print("0bias：男比女选“是的”的概率")
    print(np.average(np.array(data[0]), axis = 0))
    if name == "wx_pcf_divsum_all_data":
        print("归一化是除10个词")
        print(np.sum(np.average(np.array(data[0]), axis = 0), axis = 0))
    print("1bias：女比男选“是的”的概率")
    print(np.average(np.array(data[1]), axis = 0))
    if name == "wx_pcf_divsum_all_data":
        print(np.sum(np.average(np.array(data[1]), axis = 0), axis = 0))
    print("totalbias")
    print(np.average(np.array(array), axis = 0))
    if name == "wx_pcf_divsum_all_data":
        print(np.sum(np.average(array, axis = 0), axis = 0))
    print("---{}---end".format(name))

def main():

    models_and_dataset = [
        'CPM-prob_formats_1'
    ]
    verbalizer = [
        ['是的','不是'],
        ['是','否'],
        ['对','错'],
        ['同意','反对'],
        ['正确','错误']
    ]

    for model_and_dataset in models_and_dataset:
        for k in range(2):   

            print("--------------------Prompt-----------------------------------")
            if k == 0:
                print("无")
            else:
                print("问题：\n 答案")

            raw_data = json.load(open(os.path.join(model_and_dataset,"prob_formats_{}{}.json".format(k,k)),'r'))
            raw_divsum_data = []
            wx_pcf_data = []
            wx_pcf_divsum_data = []
            wx_p_average_data = []
            wx_p_average_divsum_data = []
            wx_pcf_divsum_all_data = []

            num_classes = len(verbalizer)*2
            p_cf = json.load(open("../Prob_cf/CPM_p{}.json".format(k),'r'))
            p_cf = np.array(p_cf).reshape(num_classes,1).squeeze(1)
            W = np.linalg.inv(np.identity(num_classes) * p_cf)

            p_average = np.average(np.array(raw_data),axis = 0)
            p_average = p_average.reshape(num_classes,1).squeeze(1)
            W_average = np.linalg.inv(np.identity(num_classes) * p_average)

            for i, group in enumerate(raw_data):

                raw_divsum_data.append(divsum(group))
                reshape_group = np.array(group).reshape(num_classes, 1)
                wx_pcf = np.matmul(W , reshape_group).reshape(num_classes//2, 2)
                wx_pcf_data.append(wx_pcf.tolist())

                wx_pcf_divsum_data.append(divsum(wx_pcf.tolist()))
                wx_pcf_divsum_all_data.append(divsum_all(wx_pcf.tolist()))

                wx_p_average = np.matmul(W_average, reshape_group).reshape(num_classes//2, 2)
                wx_p_average_data.append(wx_p_average.tolist())
                wx_p_average_divsum_data.append(divsum(wx_p_average.tolist()))
            
            output(raw_data, 'raw_data')
            output(raw_divsum_data, 'raw_divsum_data')
            output(wx_pcf_data, 'wx_pcf_data')
            output(wx_pcf_divsum_data, 'wx_pcf_divsum_data')
            output(wx_p_average_data, 'wx_p_average_data')
            output(wx_p_average_divsum_data, 'wx_p_average_divsum_data')
            output(wx_pcf_divsum_all_data, 'wx_pcf_divsum_all_data')
            print("--------------------PromptEnd-----------------------------------")
# for model_and_dataset in models_and_dataset:
#     print("----------------start-------------")
#     for k in range(2):
#         data = json.load(open(os.path.join(model_and_dataset,"prob_formats_{}{}_divsum.json".format(k,k)),'r'))
#         average_prob = np.average(np.array(data),axis = 0)
#         # average_prob = json.load(open("../Prob_cf/CPM_p{}.json".format(k),'r'))
#         # average_prob = np.array(average_prob)
#         new_data = [[],[]]
#         for i, group in enumerate(data):
#             new_data[i%2].append((np.array(group) - average_prob).tolist())
#         # json.dump(new_data, open(os.path.join(model_and_dataset,"prob_formats_{}{}_minus_Probcf.json".format(k,k)),'w'))
        
#         x = "无" if k == 0 else "问题：   答案："
#         print(verbalizer,"\n", x,"\n",np.average(np.array(new_data[0]), axis = 0)-np.average(np.array(new_data[1]), axis = 0))
#         # print(verbalizer,"\n", np.average(np.array(new_data[1]), axis = 0))
#         # x = [
#         #     [
#         #         [1,2],
#         #         [3,4]
#         #     ],
#         #     [
#         #         [5,6],
#         #         [7,8]
#         #     ]
#         # ]
#         # print(np.average(np.array(x), axis = 0))
#     print("----------------end---------------")

#     for model_and_dataset in models_and_dataset:
#         print("----------------start-------------")
#         for k in range(2):
#             f = open(os.path.join(model_and_dataset,"prob_formats_{}.csv".format(k)),'w', encoding = 'utf-8')
#             for jj in range(5):
#                 for kk in range(2):
#                     f.write(verbalizer[jj][kk]+"1,"+verbalizer[jj][kk]+"2,")
#             f.write("\n")
#             data = json.load(open(os.path.join(model_and_dataset,"prob_formats_{}{}_divsum.json".format(k,k)),'r'))
#             average_prob = np.average(np.array(data),axis = 0)
#             # average_prob = json.load(open("../Prob_cf/CPM_p{}.json".format(k),'r'))
#             # average_prob = np.array(average_prob)
#             new_data = [[],[]]
#             for i, group in enumerate(data):
#                 new_data[i%2].append((np.array(group) - average_prob).tolist())
#                 if i%2 == 1:
#                     for jj in range(5):
#                         for kk in range(2):
#                             f.write(str(data[i-1][jj][kk])+","+str(data[i][jj][kk])+",")
#                     f.write("\n")
#             # json.dump(new_data, open(os.path.join(model_and_dataset,"prob_formats_{}{}_minus_average.json".format(k,k)),'w'))
            
#             x = "无" if k == 0 else "问题：   答案："
#             print(verbalizer,"\n", x,"\n",np.average(np.array(new_data[0]), axis = 0)-np.average(np.array(new_data[1]), axis = 0))
#             # print(verbalizer,"\n", np.average(np.array(new_data[1]), axis = 0))
#             # x = [
#             #     [
#             #         [1,2],
#             #         [3,4]
#             #     ],
#             #     [
#             #         [5,6],
#             #         [7,8]
#             #     ]
#             # ]
#             # print(np.average(np.array(x), axis = 0))
#         print("----------------end---------------")

if __name__ == "__main__":
    # x = np.array([
    #     [1,2],
    #     [3,4]
    # ])
    # print(x.reshape(4,1))
    # print(x.reshape(4,1).reshape(2,2))
    main()