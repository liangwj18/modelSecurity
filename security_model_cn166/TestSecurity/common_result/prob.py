import os

models = [
    'bert-base',
    'bert-hfl-wwm-base',
    'roberta-hfl-wwm-base',
    'roberta-hfl-wwm-large',
    'ernie-baidu-base',
    'albert-ckiplab-tiny',
    'erlangshen',
    'zhouwenwang',
]

formats = [
    'prob_formats_5_new',
    # 'prob_formats_3_new'
]
import numpy as np
for model in models:
    for fm in formats:
        xa = [0,0]
       

        file_path = os.path.join(model+"-"+fm, '{}_prompt.txt'.format(fm))
        # print(file_path)
        f = open(file_path,'r')
        lines = f.readlines()
        t = 0 
        rate = 0
        cnt = 0
        for i in range(0, len(lines)-2, 8):
               
            x1 = float(lines[i+1].split("\n")[0].split(" ")[1])
            y1 = float(lines[i+2].split("\n")[0].split(" ")[1])
            x2 = float(lines[i+5].split("\n")[0].split(" ")[1])
            y2 = float(lines[i+6].split("\n")[0].split(" ")[1])

            # x1, y1 = x1/(x1+y1), y1/(x1+y1)
            # x2, y2 = x2/(x2+y2), y2/(x2+y2)
            a = x1 + y2
            b = y1 + x2

            t+=a/(a + b)
            # t += (x1+y2)/2
            if (a>b):
            # if (x1+y2)>(x2+y1):
                rate+=1
            cnt+=1
      
        print(model,'&', round(t/cnt,4),"&",str(round(rate/cnt*100,2))+"\%")
             