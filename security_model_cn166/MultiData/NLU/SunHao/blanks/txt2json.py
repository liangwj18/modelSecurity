import json
import os

for file in os.walk("./"):
    for file_name in file[2]:
        if file_name.endswith("txt"):
            f = open(file_name, 'r')
            lines = f.readlines()
            x = []
            for line in lines:
                x.append(line.split("\n")[0])
            json.dump(x, open(file_name.split(".")[0]+".json", 'w'))