import json
import numpy as np
import os
def main():

    models_and_dataset = [
        'Eva-generate_formats_1',
        'Eva-generate_formats_2'
    ]
    ff = open("agreement_response.txt",'w')
    for model_and_dataset in models_and_dataset:
        file = os.path.join(model_and_dataset, model_and_dataset.split("-")[1]+".txt")
        print(file)
        
        with open(file, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)-1):
                # print(lines[i])
                response = lines[i].split(" ")[1]
                ff.write(response+"\n")

main()