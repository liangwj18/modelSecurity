# import json
# with open("./religion.txt", 'r') as f:
#     lines = f.readlines()
#     a = []
#     b = []
#     for line in lines:
#         word = line.split("\n")[0].split("|")[1]
#         print(word)
#     #     if len(word) >= 2:
#     #         a.append("曾经是一名"+word)
#     #         b.append("永远无法成为一名"+word)
#     # json.dump(a,open("./gender_p_act.json",'w'))
#     # json.dump(b,open("./gender_p_act_neg.json",'w'))
import json
with open("./education_good.txt", 'r') as f:
    lines = f.readlines()
    a = []
    b = []
    for line in lines:
        word = line.split("\n")[0]
        # print(word)
        if len(word) >= 2:
            a.append(word)
    json.dump(a,open("./education_p_act_neg.json",'w'))
    # json.dump(b,open("./gender_p_act_neg.json",'w'))


with open("./education_bad.txt", 'r') as f:
    lines = f.readlines()
    a = []
    b = []
    for line in lines:
        word = line.split("\n")[0]
        # print(word)
        if len(word) >= 2:
            a.append(word)
    json.dump(a,open("./education_p_act.json",'w'))