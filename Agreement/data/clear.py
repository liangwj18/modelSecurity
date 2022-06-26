import json
subj = [
"小明",
"小军",
"小强",
"明明",
"军军",
"强强",
"大明",
"大军",
"大强",
"小红",
"小雯",
"小冰",
"红红",
"雯雯",
"冰冰",
"红儿",
"雯儿",
"冰儿",
]

with open("round1_nolabel_clean.csv",'w') as ff:
    with open("round1_nolabel.csv", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\n")[0]
            if len(line) > 80:
                continue
            if len(line.split("[SEP]")) < 2:
                continue
            ffff = True
            for k in subj:
                if len(line.split(k)) >= 2:
                    ffff = False
                    break
            if ffff:
                ff.write(line+"\n")